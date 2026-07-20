from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from virtual_staining.config import ConfigError, config_hash, load_config, save_effective_config


def test_config_precedence_and_hash(tmp_path: Path) -> None:
    config = load_config(
        Path("configs/smoke.yaml"),
        ["train.batch_size=3", "data.targets=[CD68]"],
        include_resolved=False,
    )
    assert config["model"]["name"] == "residual_unet"
    assert config["train"]["batch_size"] == 3
    assert config_hash(config) == config_hash(config)
    output = save_effective_config(config, tmp_path / "effective.yaml")
    assert output.is_file()


def test_initial_round_config_is_roi_jpg_safe_and_space_bounded() -> None:
    config = load_config(
        Path("configs/initial_round_cd68.yaml"), include_resolved=False
    )

    assert config["project"]["output_root"] == "outputs/initial_round"
    assert config["data"]["root"] == "AUTO"
    assert config["data"]["targets"] == ["CD68"]
    assert config["data"]["submit_targets"] == ["CD68"]
    assert config["model"]["context"]["enabled"] is False
    assert config["validation"]["primary_domain"] == "jpg"
    assert config["validation"]["domains"] == ["float", "uint8", "jpg"]
    assert config["validation"]["group_by_roi"] is True
    assert config["validation"]["bootstrap_by_roi"] is True
    assert config["train"]["save_top_k"] == 1
    assert config["validation"]["save_predictions"] is False


def test_config_rejects_unknown_cli_override() -> None:
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(
            Path("configs/smoke.yaml"),
            ["model.base_channnels=12"],
            include_resolved=False,
        )


def test_context_cache_size_is_strictly_validated() -> None:
    config = load_config(
        Path("configs/default.yaml"),
        ["model.context.cache_size=7"],
        include_resolved=False,
    )
    assert config["model"]["context"]["cache_size"] == 7

    for invalid in ("-1", "1.5", "true"):
        with pytest.raises(ConfigError, match="cache_size"):
            load_config(
                Path("configs/default.yaml"),
                [f"model.context.cache_size={invalid}"],
                include_resolved=False,
            )


def test_p0_a4_is_an_exact_cross_attention_extension_of_a3() -> None:
    a3_config = load_config(
        Path("configs/performance_v2/p0_a3_context.yaml"),
        include_resolved=False,
    )
    a4_config = load_config(
        Path("configs/performance_v2/p0_a4_cross_attention.yaml"),
        include_resolved=False,
    )
    expected_a4 = deepcopy(a3_config)
    expected_a4["model"]["context"]["bottleneck_cross_attention"] = True

    assert a3_config["model"]["name"] == "multi_marker_restorer"
    assert a4_config["model"]["name"] == "multi_marker_restorer"
    assert a3_config["model"]["context"]["bottleneck_cross_attention"] is False
    assert a4_config == expected_a4

    suite_path = Path("configs/performance_v2/ablation/p0.yaml")
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    a4_stage = suite["stages"]["A4"]
    assert a4_stage["parent"] == "A3"
    assert a4_stage["config"] == (
        "configs/performance_v2/p0_a4_cross_attention.yaml"
    )
    assert a4_stage["changes"] == ["context_cross_attention"]
    assert a4_stage["requires_verified_grid"] is True


def test_p1_data_features_are_schema_validated_and_default_off() -> None:
    config = load_config(Path("configs/default.yaml"), include_resolved=False)
    assert config["data"]["grouped_inner_folds"]["enabled"] is False
    assert config["data"]["activity_sampler"]["enabled"] is False

    for override in (
        "data.grouped_inner_folds.fold_count=1",
        "data.grouped_inner_folds.require_official_train=false",
        "data.grouped_inner_folds.coordinate_source=edge_graph",
        "data.activity_sampler.num_bins=0",
        "data.activity_sampler.seed=-1",
    ):
        extra = []
        if "grouped_inner_folds.require" in override or "coordinate_source" in override:
            extra.append("data.grouped_inner_folds.enabled=true")
        with pytest.raises(ConfigError):
            load_config(
                Path("configs/default.yaml"),
                [*extra, override],
                include_resolved=False,
            )


def test_prototype_attention_visuals_are_strictly_flagged_and_default_off() -> None:
    default = load_config(Path("configs/default.yaml"), include_resolved=False)
    monitor = default["train"]["prototype_monitor"]
    assert monitor["attention_visuals_enabled"] is False
    assert monitor["attention_visual_count"] == 4
    assert monitor["attention_visual_seed"] == 2026
    assert monitor["attention_visual_size"] == 256

    enabled = load_config(
        Path("configs/default.yaml"),
        [
            "train.prototype_monitor.enabled=true",
            "train.prototype_monitor.attention_visuals_enabled=true",
        ],
        include_resolved=False,
    )
    assert enabled["train"]["prototype_monitor"]["attention_visuals_enabled"] is True

    with pytest.raises(ConfigError, match="monitor.enabled"):
        load_config(
            Path("configs/default.yaml"),
            ["train.prototype_monitor.attention_visuals_enabled=true"],
            include_resolved=False,
        )
    for override in (
        "train.prototype_monitor.attention_visual_count=0",
        "train.prototype_monitor.attention_visual_seed=-1",
        "train.prototype_monitor.attention_visual_size=4",
    ):
        with pytest.raises(ConfigError):
            load_config(
                Path("configs/default.yaml"),
                [override],
                include_resolved=False,
            )


def test_ensemble_provenance_guards_are_schema_validated() -> None:
    config = load_config(
        Path("configs/performance_v2/ensemble.yaml"), include_resolved=False
    )
    assert config["ensemble"]["validation_only"] is True
    assert config["ensemble"]["cross_validate_weights"] is True
    assert config["ensemble"]["optimizer"] == "coordinate"
    assert config["ensemble"]["optimizer_failure_policy"] == "uniform"
    assert config["ensemble"]["allow_unsafe_model_soup_lineage"] is False
    assert config["ensemble"]["allow_unsafe_model_soup_validation"] is False

    with pytest.raises(ConfigError, match="validation/OOF-only"):
        load_config(
            Path("configs/performance_v2/ensemble.yaml"),
            ["ensemble.validation_only=false"],
            include_resolved=False,
        )
    with pytest.raises(ConfigError, match="cross_validate_weights"):
        load_config(
            Path("configs/performance_v2/ensemble.yaml"),
            ["ensemble.cross_validate_weights=not-a-boolean"],
            include_resolved=False,
        )
    for override in (
        "ensemble.optimizer=unknown",
        "ensemble.optimizer_failure_policy=ignore",
    ):
        with pytest.raises(ConfigError, match="ensemble.optimizer"):
            load_config(
                Path("configs/performance_v2/ensemble.yaml"),
                [override],
                include_resolved=False,
            )
    with pytest.raises(ConfigError, match="allow_unsafe_model_soup_lineage"):
        load_config(
            Path("configs/performance_v2/ensemble.yaml"),
            ["ensemble.allow_unsafe_model_soup_lineage=not-a-boolean"],
            include_resolved=False,
        )
    with pytest.raises(ConfigError, match="allow_unsafe_model_soup_validation"):
        load_config(
            Path("configs/performance_v2/ensemble.yaml"),
            ["ensemble.allow_unsafe_model_soup_validation=not-a-boolean"],
            include_resolved=False,
        )
