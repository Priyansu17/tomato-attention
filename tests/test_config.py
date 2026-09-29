"""Tests for the configuration loader and the authoritative label order."""

from __future__ import annotations

import copy

import pytest

from src.config import (
    REQUIRED_KEYS,
    REQUIRED_SECTIONS,
    ConfigError,
    cfg_get,
    class_label_map,
    config_fingerprint,
    load_config,
    validate_config,
)

# Deliberately chosen so that sorted() order differs from the configured order:
# alphabetical order would be Bacterial_spot, Early_blight, healthy.
ORDER_PROBE_CLASSES = ("Tomato___healthy", "Tomato___Early_blight", "Tomato___Bacterial_spot")


def test_real_config_loads_with_all_sections(real_config):
    for section in REQUIRED_SECTIONS:
        assert section in real_config
    assert len(REQUIRED_KEYS) >= 30


def test_real_config_key_defaults(real_config):
    assert cfg_get(real_config, "data.image_size") == 224
    assert cfg_get(real_config, "data.num_classes") == 10
    assert cfg_get(real_config, "data.loader.batch_size") == 16
    assert cfg_get(real_config, "training.grad_accum_steps") == 2
    assert cfg_get(real_config, "training.class_weights") == "none"
    assert cfg_get(real_config, "attention.cbam.channel.reduction_ratio") == 16
    assert cfg_get(real_config, "attention.cbam.spatial.kernel_size") == 7
    assert cfg_get(real_config, "gradcam.target") == "post_attention"
    assert cfg_get(real_config, "paths.manifests_dir") == "results/manifests"


def test_dotted_get_and_default():
    assert cfg_get({"a": {"b": {"c": 1}}}, "a.b.c") == 1
    assert cfg_get({"a": {}}, "a.b.c", "fallback") == "fallback"
    assert cfg_get({}, "a", None) is None


def test_dotted_overrides_are_applied(config_factory):
    cfg = config_factory(ORDER_PROBE_CLASSES)
    assert cfg_get(cfg, "data.num_classes") == 3
    assert cfg_get(cfg, "data.class_names") == list(ORDER_PROBE_CLASSES)


def test_class_label_map_follows_config_order(real_config):
    """QA2: label ids come only from the config order, never from sorting."""
    labels = class_label_map(real_config)
    assert labels["Tomato___Bacterial_spot"] == 0
    assert labels["Tomato___healthy"] == 9
    ordered = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
    assert ordered == list(real_config["data"]["class_names"])
    case_insensitive = sorted(real_config["data"]["class_names"], key=str.lower)
    alphabetical = {name: i for i, name in enumerate(case_insensitive)}
    assert labels != alphabetical, "label ids must not come from a case-insensitive sort"
    assert labels == {name: i for i, name in enumerate(real_config["data"]["class_names"])}


def test_label_order_probe_is_meaningful_and_uses_config_order(config_factory):
    """The probe class list must sort differently from the configured order."""
    cfg = config_factory(ORDER_PROBE_CLASSES)
    labels = class_label_map(cfg)
    assert sorted(ORDER_PROBE_CLASSES) != list(ORDER_PROBE_CLASSES)
    assert labels == {
        "Tomato___healthy": 0,
        "Tomato___Early_blight": 1,
        "Tomato___Bacterial_spot": 2,
    }


def test_config_fingerprint_is_stable_and_value_sensitive(real_config):
    assert config_fingerprint(real_config) == config_fingerprint(copy.deepcopy(real_config))
    changed = copy.deepcopy(real_config)
    changed["seed"]["value"] = changed["seed"]["value"] + 1
    assert config_fingerprint(changed) != config_fingerprint(real_config)


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_missing_section_raises(real_config):
    broken = copy.deepcopy(real_config)
    broken.pop("gradcam")
    with pytest.raises(ConfigError, match="sections"):
        validate_config(broken)


def test_missing_key_raises(real_config):
    broken = copy.deepcopy(real_config)
    broken["paths"].pop("manifests_dir")
    with pytest.raises(ConfigError, match="missing required config keys"):
        validate_config(broken)


def test_split_ratios_must_sum_to_one(real_config):
    broken = copy.deepcopy(real_config)
    broken["data"]["split"]["train"] = 0.5
    with pytest.raises(ConfigError, match="sum to 1.0"):
        validate_config(broken)


def test_class_names_length_must_match_num_classes(real_config):
    broken = copy.deepcopy(real_config)
    broken["data"]["class_names"] = broken["data"]["class_names"][:-1]
    with pytest.raises(ConfigError, match="num_classes"):
        validate_config(broken)


def test_invalid_attention_values_raise(real_config):
    for key, value, message in (
        ("reduction_ratio", 0, "reduction_ratio"),
        ("reduction_ratio", True, "reduction_ratio"),
    ):
        broken = copy.deepcopy(real_config)
        broken["attention"]["cbam"]["channel"][key] = value
        with pytest.raises(ConfigError, match=message):
            validate_config(broken)
    broken = copy.deepcopy(real_config)
    broken["attention"]["cbam"]["spatial"]["kernel_size"] = 4
    with pytest.raises(ConfigError, match="kernel_size"):
        validate_config(broken)
    broken = copy.deepcopy(real_config)
    broken["attention"]["type"] = "transformer"
    with pytest.raises(ConfigError, match="attention.type"):
        validate_config(broken)


def test_smoke_test_cannot_exceed_one_epoch(real_config):
    broken = copy.deepcopy(real_config)
    broken["smoke_test"]["max_epochs"] = 2
    with pytest.raises(ConfigError, match="smoke_test.max_epochs"):
        validate_config(broken)


def test_duplicate_class_names_raise(real_config):
    broken = copy.deepcopy(real_config)
    broken["data"]["class_names"] = ["A", "A"]
    broken["data"]["num_classes"] = 2
    with pytest.raises(ConfigError, match="duplicate"):
        class_label_map(broken)
