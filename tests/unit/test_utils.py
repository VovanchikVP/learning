from utils import cer, is_val_sample, levenshtein


def test_levenshtein():
    assert levenshtein("кот", "кот") == 0
    assert levenshtein("кот", "код") == 1
    assert levenshtein("", "код") == 3


def test_cer():
    assert cer("кот", "кот") == 0.0
    assert cer("ко", "кот") == 1 / 3
    assert cer("что-то", "") == 1.0
    assert cer("", "") == 0.0


def test_val_split_is_deterministic_and_proportional():
    ids = [f"id-{i}" for i in range(2000)]
    first = [is_val_sample(i, 0.1) for i in ids]
    assert first == [is_val_sample(i, 0.1) for i in ids]  # устойчиво между запусками
    share = sum(first) / len(ids)
    assert 0.07 < share < 0.13


def test_zero_fraction_disables_validation():
    assert not any(is_val_sample(f"id-{i}", 0.0) for i in range(100))
