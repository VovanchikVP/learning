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
from torch.utils.data import DataLoader

from config import Settings
from constants import training as tc
from constants.networks import BLANK_INDEX
from constants.defaults import DEVICE_AUTO
from constants import storage
from constants.storage import CHECKPOINT_SUFFIX, TMP_SUFFIX
from schemas import EvalResult, GroupName, GroupStatus, GroupTrainingState, TrainKind, TrainResult
from templates import errors, logs, notes
from utils import cer as char_error_rate
from .charset import CharSet
from .dataset import OCRDataset, make_collate, preprocess
from .networks import CRNN, greedy_decode, output_length
from .store import Sample, SampleStore

log = logging.getLogger(__name__)


def pick_device(name: str) -> torch.device:
    if name and name != DEVICE_AUTO:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class GroupTrainer:
    """Модель одной группы (слова или цифры): обучение, дообучение, инференс."""

    def __init__(self, group: GroupName, store: SampleStore, settings: Settings) -> None:
        self.group = group
        self.store = store
        self.settings = settings
        self.device = pick_device(settings.device)
        # CTC на MPS реализован не везде — считаем лосс на CPU, автоград это переживает
        self.ctc_on_cpu = self.device.type == "mps"
        self.charset = CharSet()
        self.model: CRNN | None = None
        self.state = GroupTrainingState()
        self.lock = threading.RLock()
        self.training = False
        self.ckpt_path = settings.checkpoint_dir / f"{group}{CHECKPOINT_SUFFIX}"
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self._ctc = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)
        self.load_checkpoint()

    # ------------------------------------------------------------------ модель

    @property
    def is_ready(self) -> bool:
        return self.model is not None and self.state.version > 0

    def _ensure_model(self, texts: list[str]) -> list[str]:
        """Создаёт модель или расширяет алфавит под новые символы."""
        new_chars = self.charset.extend(texts)
        if self.model is None:
            self.model = CRNN(self.charset.size, self.settings.img_height).to(self.device)
        elif new_chars:
            log.info(logs.ALPHABET_EXTENDED, self.group, "".join(new_chars))
            self.model.expand_classes(self.charset.size)
            self.model.to(self.device)
        return new_chars

    def _loader(self, samples: list[Sample], batch_size: int, augment: bool, shuffle: bool) -> DataLoader:
        ds = OCRDataset(
            samples,
            self.charset,
            self.settings.img_height,
            self.settings.max_img_width,
            augment=augment,
            seed=self.state.version,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.settings.num_workers,
            collate_fn=make_collate(self.charset),
            drop_last=False,
        )

    # --------------------------------------------------------------- обучение

    def _ctc_loss(self, logits: torch.Tensor, batch) -> torch.Tensor:
        log_probs = logits.log_softmax(dim=2)
        if self.ctc_on_cpu:
            log_probs = log_probs.cpu()
        return self._ctc(log_probs, batch.targets, batch.input_lengths, batch.target_lengths)

    def _train_epochs(
        self,
        train_samples: list[Sample],
        val_samples: list[Sample],
        epochs: int,
        lr: float,
        patience: int,
    ) -> tuple[float, EvalResult]:
        assert self.model is not None
        model = self.model
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            factor=tc.SCHEDULER_FACTOR,
            patience=max(tc.SCHEDULER_MIN_PATIENCE, patience // tc.SCHEDULER_PATIENCE_DIVISOR),
        )
        loader = self._loader(train_samples, self.settings.batch_size, augment=True, shuffle=True)
        # без валидации (совсем маленький датасет) следим за качеством по части train
        eval_samples = val_samples or train_samples[: tc.FALLBACK_EVAL_SAMPLES]

        best = self.evaluate(eval_samples)
        best_state = copy.deepcopy(model.state_dict())
        bad_epochs = 0
        last_loss = float("nan")

        for epoch in range(1, epochs + 1):
            model.train()
            total, seen = 0.0, 0
            for batch in loader:
                images = batch.images.to(self.device, non_blocking=True)
                logits = model(images)
                loss = self._ctc_loss(logits, batch)
                if not torch.isfinite(loss):
                    log.warning(logs.INVALID_LOSS, self.group)
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tc.GRAD_CLIP_NORM)
                opt.step()
                total += loss.detach().item() * len(batch.texts)
                seen += len(batch.texts)
            last_loss = total / max(seen, 1)
            metrics = self.evaluate(eval_samples)
            if metrics.cer is not None:
                sched.step(metrics.cer)
            log.info(
                logs.EPOCH, self.group, epoch, epochs, last_loss,
                logs.METRIC_NOT_AVAILABLE if metrics.cer is None else f"{metrics.cer:.4f}",
                metrics.accuracy,
            )
            improved = best.cer is None or (
                metrics.cer is not None and metrics.cer < best.cer - tc.CER_IMPROVEMENT_EPS
            )
            if improved:
                best = metrics
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    log.info(logs.EARLY_STOP, self.group, epoch)
                    break

        model.load_state_dict(best_state)
        return last_loss, best

    def train_full(self) -> TrainResult:
        """Полное обучение с нуля по всем накопленным данным."""
        started = time.time()
        with self.lock:
            self.training = True
            try:
                train_samples, val_samples = self.store.split()
                if len(train_samples) < self.settings.min_samples_to_train:
                    return TrainResult(
                        group=self.group, kind=TrainKind.FULL, started_at=started, accepted=False,
                        train_samples=len(train_samples),
                        note=notes.NOT_ENOUGH_DATA.format(
                            count=len(train_samples), minimum=self.settings.min_samples_to_train
                        ),
                    )
                self.charset = CharSet()
                self.model = None
                self._ensure_model([s.text for s in self.store.all()])
                loss, metrics = self._train_epochs(
                    train_samples, val_samples,
                    self.settings.initial_epochs, self.settings.lr,
                    self.settings.early_stop_patience,
                )
                self._promote(metrics)
                result = TrainResult(
                    group=self.group, kind=TrainKind.FULL, started_at=started,
                    duration_sec=time.time() - started, epochs=self.settings.initial_epochs,
                    train_samples=len(train_samples), val_samples=len(val_samples),
                    train_loss=loss, val_cer=metrics.cer, val_accuracy=metrics.accuracy,
                    accepted=True,
                )
                self._record(result)
                self.save_checkpoint()
                return result
            finally:
                self.training = False

    def train_incremental(self, new_ids: list[str]) -> TrainResult:
        """Дообучение на новой порции с подмешиванием старых примеров (replay).

        Если валидационный CER ухудшился сильнее допуска, изменения откатываются —
        непрерывное обучение не должно портить уже работающую модель.
        """
        started = time.time()
        with self.lock:
            if not self.is_ready:
                return self.train_full()
            self.training = True
            try:
                train_samples, val_samples = self.store.split()
                new_set = set(new_ids)
                fresh = [s for s in train_samples if s.id in new_set]
                old = [s for s in train_samples if s.id not in new_set]
                if not fresh:
                    return TrainResult(
                        group=self.group, kind=TrainKind.INCREMENTAL, started_at=started,
                        accepted=False, note=notes.ALL_NEW_IN_VALIDATION,
                    )
                rng = random.Random(self.state.version)
                replay_n = min(len(old), int(len(fresh) * self.settings.replay_ratio))
                replay = rng.sample(old, replay_n) if replay_n else []
                batch_samples = fresh + replay
                rng.shuffle(batch_samples)

                self._ensure_model([s.text for s in fresh])
                baseline = self.evaluate(val_samples)
                snapshot = copy.deepcopy(self.model.state_dict())

                loss, metrics = self._train_epochs(
                    batch_samples, val_samples,
                    self.settings.incremental_epochs, self.settings.incremental_lr,
                    max(tc.INCREMENTAL_MIN_PATIENCE,
                        self.settings.early_stop_patience // tc.INCREMENTAL_PATIENCE_DIVISOR),
                )

                accepted = self._is_acceptable(metrics, baseline)
                note = ""
                if accepted:
                    self._promote(metrics)
                else:
                    self.model.load_state_dict(snapshot)
                    note = notes.ROLLBACK.format(
                        cer=metrics.cer,
                        baseline=baseline.cer,
                        tolerance=self.settings.cer_regression_tolerance,
                    )
                    log.warning(logs.ROLLED_BACK, self.group, note)

                result = TrainResult(
                    group=self.group, kind=TrainKind.INCREMENTAL, started_at=started,
                    duration_sec=time.time() - started, epochs=self.settings.incremental_epochs,
                    train_samples=len(batch_samples), new_samples=len(fresh),
                    val_samples=len(val_samples), train_loss=loss,
                    val_cer=metrics.cer, val_accuracy=metrics.accuracy,
                    baseline_cer=baseline.cer, accepted=accepted, note=note,
                )
                self._record(result)
                if accepted:
                    self.save_checkpoint()
                return result
            finally:
                self.training = False

    def _is_acceptable(self, metrics: EvalResult, baseline: EvalResult) -> bool:
        if metrics.cer is None or baseline.cer is None:
            return True  # сравнивать не с чем
        return metrics.cer <= baseline.cer + self.settings.cer_regression_tolerance

    def _promote(self, metrics: EvalResult) -> None:
        self.state.version += 1
        self.state.trained_samples = len(self.store)
        self.state.best_val_cer = metrics.cer
        self.state.best_val_accuracy = metrics.accuracy
        self.state.updated_at = time.time()

    def _record(self, result: TrainResult) -> None:
        self.state.history.append(result)
        self.state.history = self.state.history[-tc.HISTORY_LIMIT :]

    # -------------------------------------------------------------- инференс

    @torch.no_grad()
    def evaluate(self, samples: list[Sample]) -> EvalResult:
        if not samples or self.model is None:
            return EvalResult()
        self.model.eval()
        loader = self._loader(
            samples, max(tc.MIN_EVAL_BATCH_SIZE, self.settings.batch_size), augment=False, shuffle=False
        )
        total_cer, exact, count = 0.0, 0, 0
        for batch in loader:
            logits = self.model(batch.images.to(self.device))
            texts, _ = greedy_decode(logits.cpu(), batch.input_lengths, self.charset)
            for pred, target in zip(texts, batch.texts, strict=True):
                total_cer += char_error_rate(pred, target)
                exact += int(pred == target)
                count += 1
        return EvalResult(cer=total_cer / count, accuracy=exact / count, count=count)

    @torch.no_grad()
    def predict(self, images: list[Image.Image]) -> list[tuple[str, float]]:
        if not self.is_ready:
            raise RuntimeError(errors.GROUP_MODEL_NOT_READY.format(group=self.group))
        with self.lock:
            self.model.eval()
            arrays = [
                preprocess(img, self.settings.img_height, self.settings.max_img_width)
                for img in images
            ]
            max_w = max(a.shape[2] for a in arrays)
            padded = []
            for arr in arrays:
                if arr.shape[2] < max_w:
                    pad_value = float(arr[:, :, -1].mean())
                    arr = np.pad(
                        arr, ((0, 0), (0, 0), (0, max_w - arr.shape[2])), constant_values=pad_value
                    )
                padded.append(arr)
            tensor = torch.from_numpy(np.stack(padded)).to(self.device)
            lengths = torch.full((len(padded),), output_length(max_w), dtype=torch.long)
            logits = self.model(tensor).cpu()
            texts, confidences = greedy_decode(logits, lengths, self.charset)
            return list(zip(texts, confidences, strict=True))

    # ----------------------------------------------------------- чекпоинты

    def save_checkpoint(self) -> None:
        if self.model is None:
            return
        tmp = self.ckpt_path.with_suffix(TMP_SUFFIX)
        torch.save(
            {
                storage.CKPT_GROUP: self.group,
                storage.CKPT_CHARS: self.charset.chars,
                storage.CKPT_IMG_HEIGHT: self.settings.img_height,
                storage.CKPT_MODEL: self.model.state_dict(),
                storage.CKPT_STATE: self.state.model_dump(mode="json"),
            },
            tmp,
        )
        tmp.replace(self.ckpt_path)
        log.info(
            logs.CHECKPOINT_SAVED, self.group, self.state.version,
            logs.METRIC_NOT_AVAILABLE
            if self.state.best_val_cer is None
            else f"{self.state.best_val_cer:.4f}",
        )

    def load_checkpoint(self) -> bool:
        if not self.ckpt_path.exists():
            return False
        payload = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.charset = CharSet(payload[storage.CKPT_CHARS])
        self.model = CRNN(
            self.charset.size, payload.get(storage.CKPT_IMG_HEIGHT, self.settings.img_height)
        )
        self.model.load_state_dict(payload[storage.CKPT_MODEL])
        self.model.to(self.device)
        self.state = GroupTrainingState.model_validate(payload.get(storage.CKPT_STATE, {}))
        log.info(
            logs.CHECKPOINT_LOADED, self.group, self.state.version,
            self.state.best_val_cer, self.state.trained_samples,
        )
        return True

    def status(self, pending: int = 0) -> GroupStatus:
        return GroupStatus(
            group=self.group,
            ready=self.is_ready,
            training=self.training,
            version=self.state.version,
            device=str(self.device),
            alphabet="".join(self.charset.chars),
            cached_samples=len(self.store),
            trained_samples=self.state.trained_samples,
            val_cer=self.state.best_val_cer,
            val_accuracy=self.state.best_val_accuracy,
            updated_at=self.state.updated_at,
            pending=pending,
            last_run=self.state.history[-1] if self.state.history else None,
        )
