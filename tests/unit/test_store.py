from training.store import SampleStore, safe_name
from tests.conftest import make_image


def test_add_is_idempotent_and_persists(tmp_path):
    store = SampleStore(tmp_path, "words", val_fraction=0.1)
    assert store.add("uuid-1", make_image("кот"), "кот")
    assert not store.add("uuid-1", make_image("кот"), "кот")  # дубль игнорируется
    assert len(store) == 1

    reopened = SampleStore(tmp_path, "words", val_fraction=0.1)
    assert len(reopened) == 1
    assert reopened.get("uuid-1").text == "кот"
    assert reopened.ids() == {"uuid-1"}


def test_split_is_stable_and_covers_everything(tmp_path):
    store = SampleStore(tmp_path, "digits", val_fraction=0.2)
    for i in range(50):
        store.add(f"uuid-{i}", make_image(str(i)), str(i))
    train, val = store.split()
    assert len(train) + len(val) == 50
    assert {s.id for s in train} & {s.id for s in val} == set()
    assert store.split()[1] == val  # разбиение не плавает


def test_safe_name_sanitizes_exotic_ids(tmp_path):
    assert safe_name("0b9d2a1e-1111-2222-3333-444455556666").endswith("6666")
    assert "/" not in safe_name("../../etc/passwd")
    store = SampleStore(tmp_path, "words", val_fraction=0.0)
    store.add("../../etc/passwd", make_image("x"), "x")
    assert store.get("../../etc/passwd").path.parent == tmp_path / "words" / "images"
