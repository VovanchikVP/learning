from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from constants import storage
from templates import logs
from utils import is_val_sample

log = logging.getLogger(__name__)


def safe_name(sample_id: str) -> str:
    """Имя файла для id. uuid проходит как есть, экзотика — через sha256."""
    if storage.SAFE_NAME_RE.match(sample_id):
        return sample_id
    return hashlib.sha256(sample_id.encode()).hexdigest()


@dataclass(slots=True, frozen=True)
class Sample:
    id: str
    path: Path
    text: str


class SampleStore:
    """Локальный кэш датасета одной группы.

    Картинки из базы декодируются один раз и кладутся на диск как PNG, метки —
    в labels.jsonl. Благодаря этому дообучение не ходит в базу за старыми
    строками и может делать много эпох по локальным файлам.
    """

    def __init__(self, root: Path, group: str, val_fraction: float) -> None:
        self.group = group
        self.root = root / group
        self.images_dir = self.root / storage.IMAGES_DIRNAME
        self.labels_path = self.root / storage.LABELS_FILENAME
        self.val_fraction = val_fraction
        self._samples: dict[str, Sample] = {}
        self._lock = threading.RLock()
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.labels_path.exists():
            return
        with self.labels_path.open(encoding=storage.ENCODING) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    log.warning(logs.STORE_BROKEN_LINE, self.group)
                    continue
                sample_id = str(row["id"])
                path = self.images_dir / f"{safe_name(sample_id)}{storage.IMAGE_EXTENSION}"
                if path.exists():
                    self._samples[sample_id] = Sample(sample_id, path, row["text"])
        log.info(logs.STORE_LOADED, self.group, len(self._samples))

    def __len__(self) -> int:
        return len(self._samples)

    def __contains__(self, sample_id: str) -> bool:
        return sample_id in self._samples

    def add(self, sample_id: str, image: Image.Image, text: str) -> bool:
        """Сохраняет пример в кэш. False, если такой id уже есть."""
        with self._lock:
            if sample_id in self._samples:
                return False
            path = self.images_dir / f"{safe_name(sample_id)}{storage.IMAGE_EXTENSION}"
            image.save(path, format=storage.IMAGE_FORMAT, optimize=False)
            with self.labels_path.open("a", encoding=storage.ENCODING) as fh:
                fh.write(json.dumps({"id": sample_id, "text": text}, ensure_ascii=False) + "\n")
            self._samples[sample_id] = Sample(sample_id, path, text)
            return True

    def get(self, sample_id: str) -> Sample | None:
        return self._samples.get(sample_id)

    def all(self) -> list[Sample]:
        with self._lock:
            return list(self._samples.values())

    def split(self) -> tuple[list[Sample], list[Sample]]:
        """Детерминированное разбиение на train/val по id."""
        train, val = [], []
        for sample in self.all():
            (val if is_val_sample(sample.id, self.val_fraction) else train).append(sample)
        return train, val

    def ids(self) -> set[str]:
        with self._lock:
            return set(self._samples)

    def texts(self) -> list[str]:
        return [s.text for s in self.all()]
