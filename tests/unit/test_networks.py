import torch

from training.charset import CharSet
from training.networks import CRNN, greedy_decode, output_length


def test_output_length_matches_forward():
    model = CRNN(n_classes=5, img_height=32)
    logits = model(torch.zeros(2, 1, 32, 128))
    assert logits.shape == (output_length(128), 2, 5)


def test_expand_classes_keeps_old_weights():
    model = CRNN(n_classes=5, img_height=32)
    old_weight = model.head.weight.detach().clone()
    model.expand_classes(8)
    assert model.head.out_features == 8
    assert torch.allclose(model.head.weight[:5], old_weight)
    model.expand_classes(3)  # сужение игнорируется
    assert model.head.out_features == 8


def test_greedy_decode_collapses_repeats_and_blanks():
    charset = CharSet("аб")
    # кадры: а, а (повтор), blank, а, б -> "ааб": повтор схлопывается, blank режет
    logits = torch.full((5, 1, 3), -10.0)
    for step, index in enumerate([1, 1, 0, 1, 2]):
        logits[step, 0, index] = 10.0
    texts, confidences = greedy_decode(logits, torch.tensor([5]), charset)
    assert texts == ["ааб"]
    assert 0.0 < confidences[0] <= 1.0


def test_greedy_decode_respects_input_lengths():
    charset = CharSet("аб")
    logits = torch.full((4, 1, 3), -10.0)
    for step, index in enumerate([1, 0, 2, 2]):
        logits[step, 0, index] = 10.0
    assert greedy_decode(logits, torch.tensor([2]), charset)[0] == ["а"]
    assert greedy_decode(logits, torch.tensor([4]), charset)[0] == ["аб"]
