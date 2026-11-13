from typing import get_args

from config import Settings
from constants import api, cli, db, defaults, demo, images, networks, storage, training
from constants.groups import (
    DIGITS,
    GROUPS,
    ROUTING_CLASSIFIER,
    ROUTING_EXPLICIT,
    ROUTING_OCR_CONFIDENCE,
    WORDS,
)
from schemas.common import GroupName, RoutingMode
from training.networks import CRNN


def test_groups_match_schema_literal():
    assert GROUPS == (WORDS, DIGITS)
    assert set(get_args(GroupName)) == set(GROUPS)


def test_routing_modes_match_schema_literal():
    assert set(get_args(RoutingMode)) == {
        ROUTING_EXPLICIT,
        ROUTING_CLASSIFIER,
        ROUTING_OCR_CONFIDENCE,
    }


def test_settings_defaults_come_from_constants():
    settings = Settings()
    assert settings.table == defaults.DEFAULT_TABLE
    assert settings.retrain_threshold == defaults.DEFAULT_RETRAIN_THRESHOLD
    assert settings.img_height == defaults.DEFAULT_IMG_HEIGHT
    assert settings.device == defaults.DEVICE_AUTO
    assert settings.workdir == defaults.DEFAULT_WORKDIR
    assert settings.api_port == defaults.DEFAULT_API_PORT


def test_storage_names_are_consistent():
    assert storage.CLASSIFIER_CHECKPOINT_NAME.endswith(storage.CHECKPOINT_SUFFIX)
    assert f".{storage.IMAGE_FORMAT.lower()}" == storage.IMAGE_EXTENSION
    assert storage.SAFE_NAME_RE.match("0b9d2a1e-1111-2222-3333-444455556666")
    assert not storage.SAFE_NAME_RE.match("../etc/passwd")


def test_identifier_regex_rejects_injection():
    assert db.IDENTIFIER_RE.match("captcha")
    assert not db.IDENTIFIER_RE.match("images; drop table users")


def test_height_divisor_matches_network():
    CRNN(n_classes=3, img_height=networks.HEIGHT_DIVISOR)  # не бросает
    try:
        CRNN(n_classes=3, img_height=networks.HEIGHT_DIVISOR // 2)
    except ValueError:
        pass
    else:
        raise AssertionError("сеть должна отвергать некратную высоту")


def test_augmentation_ranges_are_ordered():
    for low, high in (
        images.AUG_SCALE_RANGE,
        images.AUG_BRIGHTNESS_RANGE,
        images.AUG_CONTRAST_RANGE,
        demo.FONT_SIZE_RANGE,
        demo.DIGITS_LENGTH_RANGE,
    ):
        assert low < high


def test_api_routes_are_absolute():
    for route in (api.ROUTE_HEALTH, api.ROUTE_STATUS, api.ROUTE_PREDICT, api.ROUTE_TRAIN):
        assert route.startswith("/")


def test_cli_commands_are_unique():
    commands = [
        cli.COMMAND_RUN,
        cli.COMMAND_WATCH,
        cli.COMMAND_BOOTSTRAP,
        cli.COMMAND_TRAIN,
        cli.COMMAND_STATUS,
        cli.COMMAND_PREDICT,
        cli.COMMAND_SEED_DB,
        cli.COMMAND_SELFTEST,
    ]
    assert len(set(commands)) == len(commands)


def test_val_buckets_allow_configured_fraction():
    assert training.VAL_BUCKETS * Settings().val_fraction >= 1
