from constants.networks import BLANK_INDEX
from training.charset import CharSet


def test_blank_reserves_zero():
    charset = CharSet("аб")
    assert charset.size == 3  # два символа + blank
    assert charset.encode("аб") == [1, 2]
    assert BLANK_INDEX == 0


def test_extend_adds_only_new_characters():
    charset = CharSet("аб")
    assert charset.extend(["абв", "где"]) == ["в", "г", "д", "е"]
    assert charset.extend(["абв"]) == []
    assert charset.size == 7


def test_encode_skips_unknown_and_decode_ignores_blank():
    charset = CharSet("аб")
    assert charset.encode("аXб") == [1, 2]
    assert charset.decode([1, 0, 2, 99]) == "аб"
    assert not charset.covers("аXб")


def test_order_is_stable_after_extend():
    charset = CharSet("аб")
    before = charset.encode("аб")
    charset.extend(["я"])
    assert charset.encode("аб") == before
