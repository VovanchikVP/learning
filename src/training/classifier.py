from __future__ import annotations

import copy
import logging
import random
import threading
import time

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from config import Settings
from constants import training as tc
from constants.groups import GROUPS
from constants.networks import (
    CLASSIFIER_CHANNELS,
    CLASSIFIER_CLASSES,
    CLASSIFIER_INPUT_WIDTH,
)
from constants import storage
from constants.storage import CLASSIFIER_CHECKPOINT_NAME, TMP_SUFFIX
from schemas import ClassifierStatus, ClassifierTrainResult
from templates import errors, logs, notes
from .dataset import preprocess
from .store import Sample, SampleStore
from .trainer import pick_device

log = logging.getLogger(__name__)


class _ClsDataset(Dataset):
    def __init__(self, items: list[tuple[Sample, int]], height: int, augment: bool, seed: int) -> None:
        self.items = items
        self.height = height
        self.augment = augment
        self._seed = seed

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sample, label = self.items[idx]
        img = Image.open(sample.path)
        img.load()
        rng = random.Random(hash((self._seed, idx, sample.id))) if self.augment else None
        return prepare(img, self.height, rng), label


def prepare(img: Image.Image, height: int, rng: random.Random | None = None) -> np.ndarray:
    """Картинка -> массив фиксированного размера [1, H, CLASSIFIER_INPUT_WIDTH]."""
    arr = preprocess(img, height, CLASSIFIER_INPUT_WIDTH, augment_rng=rng)
    w = arr.shape[2]
    if w < CLASSIFIER_INPUT_WIDTH:
        pad_value = float(arr[:, :, -1].mean())
        arr = np.pad(
            arr, ((0, 0), (0, 0), (0, CLASSIFIER_INPUT_WIDTH - w)), constant_values=pad_value
        )
    return arr


def _collate(items):
    arrays = torch.from_numpy(np.stack([a for a, _ in items]))
    labels = torch.tensor([lbl for _, lbl in items], dtype=torch.long)
    return arrays, labels


class _Net(nn.Module):
    def __init__(self, n_classes: int = CLASSIFIER_CLASSES) -> None:
        super().__init__()
        c1, c2, c3 = CLASSIFIER_CHANNELS
        self.body = nn.Sequential(
            nn.Conv2d(1, c1, 3, 1, 1), nn.BatchNorm2d(c1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, 1, 1), nn.BatchNorm2d(c2), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.ReLU(True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(c3, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x).flatten(1))


class GroupClassifier:
    """Маленький CNN-классификатор «слово или число».

    Нужен, чтобы /predict без явной группы выбирал модель надёжно: сравнение
    уверенности двух CTC-моделей с непересекающимися алфавитами даёт лишь ~75%.
    Обучается на тех же кэшированных данных и обновляется после каждого
    успешного обучения любой из OCR-моделей.
    """

    def __init__(self, settings: Settings, stores: dict[str, SampleStore]) -> None:
        self.settings = settings
        self.stores = stores
        self.device = pick_device(settings.device)
        self.model: _Net | None = None
        self.lock = threading.RLock()
        self.training = False
        self.version = 0
        self.val_accuracy = 0.0
        self.trained_samples = 0
        self.updated_at = 0.0
        self.ckpt_path = settings.checkpoint_dir / CLASSIFIER_CHECKPOINT_NAME
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.load_checkpoint()

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.version > 0

    def _items(self) -> tuple[list, list]:
        train, val = [], []
        for label, group in enumerate(GROUPS):
            g_train, g_val = self.stores[group].split()
            train.extend((s, label) for s in g_train)
            val.extend((s, label) for s in g_val)
        return train, val

    @torch.no_grad()
    def _accuracy(self, items: list) -> float:
        if not items or self.model is None:
            return 0.0
        self.model.eval()
        loader = DataLoader(
            _ClsDataset(items, self.settings.img_height, False, 0),
            batch_size=tc.CLASSIFIER_BATCH_SIZE,
            collate_fn=_collate,
            num_workers=self.settings.num_workers,
        )
        correct = 0
        for arrays, labels in loader:
            preds = self.model(arrays.to(self.device)).argmax(1).cpu()
            correct += int((preds == labels).sum())
        return correct / len(items)

    def train(self, epochs: int = tc.CLASSIFIER_EPOCHS) -> ClassifierTrainResult:
        """Переобучает классификатор на всех накопленных данных."""
        with self.lock:
            train_items, val_items = self._items()
            labels_present = {lbl for _, lbl in train_items}
            if len(labels_present) < CLASSIFIER_CLASSES or len(train_items) < tc.CLASSIFIER_MIN_SAMPLES:
                return ClassifierTrainResult(note=notes.CLASSIFIER_NEEDS_BOTH_GROUPS)
            self.training = True
            started = time.time()
            try:
                self.model = _Net().to(self.device)
                opt = torch.optim.AdamW(
                    self.model.parameters(), lr=tc.CLASSIFIER_LR, weight_decay=tc.WEIGHT_DECAY
                )
                loss_fn = nn.CrossEntropyLoss()
                loader = DataLoader(
                    _ClsDataset(train_items, self.settings.img_height, True, self.version),
                    batch_size=tc.CLASSIFIER_BATCH_SIZE, shuffle=True, collate_fn=_collate,
                    num_workers=self.settings.num_workers,
                )
                eval_items = val_items or train_items[: tc.CLASSIFIER_EVAL_FALLBACK]
                best_acc, best_state = 0.0, copy.deepcopy(self.model.state_dict())
                for epoch in range(1, epochs + 1):
                    self.model.train()
                    for arrays, labels in loader:
                        logits = self.model(arrays.to(self.device))
                        loss = loss_fn(logits, labels.to(self.device))
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    acc = self._accuracy(eval_items)
                    if acc >= best_acc:
                        best_acc, best_state = acc, copy.deepcopy(self.model.state_dict())
                    log.info(logs.CLASSIFIER_EPOCH, epoch, epochs, acc)
                    if best_acc == tc.PERFECT_ACCURACY:
                        break
                self.model.load_state_dict(best_state)
                self.version += 1
                self.val_accuracy = best_acc
                self.trained_samples = len(train_items) + len(val_items)
                self.updated_at = time.time()
                self.save_checkpoint()
                return ClassifierTrainResult(
                    trained=True,
                    version=self.version,
                    val_accuracy=best_acc,
                    samples=self.trained_samples,
                    duration_sec=time.time() - started,
                )
            finally:
                self.training = False

    @torch.no_grad()
    def predict(self, image: Image.Image) -> tuple[str, float]:
        if not self.is_ready:
            raise RuntimeError(errors.CLASSIFIER_NOT_READY)
        with self.lock:
            self.model.eval()
            arr = prepare(image, self.settings.img_height)
            logits = self.model(torch.from_numpy(arr[None]).to(self.device)).cpu()
            probs = logits.softmax(1)[0]
            idx = int(probs.argmax())
            return GROUPS[idx], float(probs[idx])

    def save_checkpoint(self) -> None:
        if self.model is None:
            return
        tmp = self.ckpt_path.with_suffix(TMP_SUFFIX)
        torch.save(
            {
                storage.CKPT_MODEL: self.model.state_dict(),
                storage.CKPT_VERSION: self.version,
                storage.CKPT_VAL_ACCURACY: self.val_accuracy,
                storage.CKPT_TRAINED_SAMPLES: self.trained_samples,
                storage.CKPT_UPDATED_AT: self.updated_at,
            },
            tmp,
        )
        tmp.replace(self.ckpt_path)

    def load_checkpoint(self) -> bool:
        if not self.ckpt_path.exists():
            return False
        payload = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.model = _Net()
        self.model.load_state_dict(payload[storage.CKPT_MODEL])
        self.model.to(self.device)
        self.version = int(payload.get(storage.CKPT_VERSION, 0))
        self.val_accuracy = float(payload.get(storage.CKPT_VAL_ACCURACY, 0.0))
        self.trained_samples = int(payload.get(storage.CKPT_TRAINED_SAMPLES, 0))
        self.updated_at = float(payload.get(storage.CKPT_UPDATED_AT, 0.0))
        log.info(logs.CLASSIFIER_CHECKPOINT_LOADED, self.version, self.val_accuracy)
        return True

    def status(self) -> ClassifierStatus:
        return ClassifierStatus(
            ready=self.is_ready,
            training=self.training,
            version=self.version,
            val_accuracy=round(self.val_accuracy, tc.METRIC_ROUND_DIGITS),
            trained_samples=self.trained_samples,
            updated_at=self.updated_at,
        )
