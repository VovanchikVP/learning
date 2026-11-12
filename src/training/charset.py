from __future__ import annotations

from collections.abc import Iterable, Sequence

from constants.networks import BLANK_INDEX


class CharSet:
    """Алфавит модели. Индекс 0 зарезервирован под CTC-blank."""

    def __init__(self, chars: Sequence[str] = ()) -> None:
        self._chars: list[str] = list(dict.fromkeys(chars))
        self._rebuild()

    def _rebuild(self) -> None:
        # индексы начинаются после blank
        self._index = {c: i + BLANK_INDEX + 1 for i, c in enumerate(self._chars)}

    @property
    def chars(self) -> list[str]:
        return list(self._chars)

    @property
    def size(self) -> int:
        """Число выходов модели: символы + blank."""
        return len(self._chars) + 1

    def extend(self, texts: Iterable[str]) -> list[str]:
        """Добавляет незнакомые символы, возвращает список новых."""
        new: list[str] = []
        for text in texts:
            for ch in text:
                if ch not in self._index and ch not in new:
                    new.append(ch)
        if new:
            self._chars.extend(new)
            self._rebuild()
        return new

    def covers(self, text: str) -> bool:
        return all(ch in self._index for ch in text)

    def encode(self, text: str) -> list[int]:
        return [self._index[ch] for ch in text if ch in self._index]

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for i in ids:
            if i == BLANK_INDEX:
                continue
            idx = int(i) - BLANK_INDEX - 1
            if 0 <= idx < len(self._chars):
                out.append(self._chars[idx])
        return "".join(out)

    def __len__(self) -> int:
        return len(self._chars)

    def __repr__(self) -> str:
        return f"CharSet(size={self.size}, chars={''.join(self._chars)!r})"
