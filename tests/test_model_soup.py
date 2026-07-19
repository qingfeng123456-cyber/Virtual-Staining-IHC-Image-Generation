from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from virtual_staining.engine.checkpoint import save_checkpoint
from virtual_staining.engine.model_soup import (
    SoupCompatibilityError,
    SoupMember,
    SoupValidationContractError,
    average_state_dicts,
    bind_soup_validation_contract,
    build_initialization_provenance,
    build_soup_validation_contract,
    checkpoint_file_sha256,
    extract_checkpoint_provenance,
    extract_checkpoint_state,
    greedy_model_soup,
    state_dict_architecture_sha256,
    validate_soup_validation_contract,
    validate_state_dict_compatibility,
)


def _state(value: float, *, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor([value, value + 2.0], dtype=dtype),
        "counter": torch.tensor(3, dtype=torch.int64),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _validation_evidence(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = _write_csv(
        tmp_path / "权威 val.csv",
        [
            {
                "canonical_key": "colon/ROI000_00_00",
                "organ": "colon",
                "split": "val",
                "source_split": "val",
                "roi_id": "ROI000",
                "roi_id_source": "filename_regex",
                "stem": "ROI000_00_00",
            },
            {
                "canonical_key": "colon/ROI001_00_00",
                "organ": "colon",
                "split": "validation",
                "source_split": "official_val",
                "roi_id": "ROI001",
                "roi_id_source": "filename_coordinate",
                "stem": "ROI001_00_00.jpg",
            },
        ],
    )
    audit_payload = {
        "total_rows": 2,
        "parsed_rows": 2,
        "parse_fraction": 1.0,
        "duplicate_coordinates": [],
        "train_val_shared_rois": [],
        "cross_split_adjacent_pairs": [],
        "boundary": {
            "direction_verified": True,
            "continuity_verified": True,
        },
        "filename_grid_verified": True,
        "context_enabled": True,
        "context_gate_reasons": [],
    }
    audit = tmp_path / "roi audit.json"
    audit.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    records = _write_csv(
        tmp_path / "per image.csv",
        [
            {
                "target": "CD68",
                "canonical_key": "colon/ROI000_00_00",
                "jpg_ssim": 0.8,
                "jpg_psnr": 28.0,
            },
            {
                "target": "CD68",
                "canonical_key": "colon/ROI001_00_00",
                "jpg_ssim": 0.81,
                "jpg_psnr": 28.5,
            },
        ],
    )
    return manifest, audit, records


def test_average_state_dicts_preserves_dtype_and_nonfloating_state() -> None:
    first = _state(0.0, dtype=torch.float16)
    second = _state(2.0, dtype=torch.float16)
    averaged = average_state_dicts([first, second], [1.0, 3.0])
    assert averaged["weight"].dtype == torch.float16
    assert torch.equal(averaged["weight"], torch.tensor([1.5, 3.5], dtype=torch.float16))
    assert torch.equal(averaged["counter"], first["counter"])
    assert averaged["counter"].data_ptr() != first["counter"].data_ptr()


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        ({"other": torch.zeros(2), "counter": torch.tensor(3)}, "keys differ"),
        (
            {"weight": torch.zeros(3), "counter": torch.tensor(3)},
            "Shape mismatch",
        ),
        (
            {"weight": torch.zeros(2, dtype=torch.float64), "counter": torch.tensor(3)},
            "Dtype mismatch",
        ),
        ({"weight": torch.zeros(2), "counter": torch.tensor(4)}, "Non-floating state"),
    ],
)
def test_state_dict_compatibility_is_strict(
    candidate: dict[str, torch.Tensor], message: str
) -> None:
    with pytest.raises(SoupCompatibilityError, match=message):
        validate_state_dict_compatibility(_state(0.0), candidate)


def test_greedy_soup_uses_fresh_validation_and_rejects_regression() -> None:
    members = [
        SoupMember("best", _state(0.0), validation_score=0.9, lineage="init-a"),
        SoupMember("helpful", _state(2.0), validation_score=0.8, lineage="init-a"),
        SoupMember("harmful", _state(20.0), validation_score=0.7, lineage="init-a"),
    ]
    evaluated: list[float] = []

    def validation(state: dict[str, torch.Tensor]) -> float:
        value = float(state["weight"][0])
        evaluated.append(value)
        return -abs(value - 1.0)

    result = greedy_model_soup(members, validation)
    assert result.member_names == ("best", "helpful")
    assert result.validation_score == pytest.approx(0.0)
    assert evaluated == pytest.approx([0.0, 1.0, 22.0 / 3.0])
    assert [trial.accepted for trial in result.trials] == [True, True, False]
    assert torch.allclose(result.state_dict["weight"], torch.tensor([1.0, 3.0]))


def test_greedy_soup_separates_weight_sources_and_optional_lineage() -> None:
    with pytest.raises(SoupCompatibilityError, match="cannot be mixed"):
        greedy_model_soup(
            [
                SoupMember("raw", _state(0.0), 1.0, weight_source="raw"),
                SoupMember("ema", _state(0.0), 0.9, weight_source="ema"),
            ],
            lambda state: float(state["weight"].mean()),
        )
    with pytest.raises(SoupCompatibilityError, match="initialization lineage"):
        greedy_model_soup(
            [
                SoupMember("one", _state(0.0), 1.0, lineage="init-a"),
                SoupMember("two", _state(0.0), 0.9, lineage="init-b"),
            ],
            lambda state: float(state["weight"].mean()),
            require_matching_lineage=True,
        )


def test_greedy_soup_requires_lineage_by_default_and_marks_unsafe_override() -> None:
    members = [
        SoupMember("one", _state(0.0), 1.0),
        SoupMember("two", _state(2.0), 0.9),
    ]
    with pytest.raises(SoupCompatibilityError, match="initialization lineage"):
        greedy_model_soup(members, lambda state: float(state["weight"].mean()))

    result = greedy_model_soup(
        members,
        lambda state: float(state["weight"].mean()),
        allow_unsafe_lineage_mismatch=True,
    )

    assert result.unsafe_lineage_override_used is True
    assert result.common_initialization_lineage is None


def test_checkpoint_provenance_is_exact_and_legacy_is_rejected(tmp_path) -> None:
    model = torch.nn.Linear(2, 2)
    pretrain_sha256 = "b" * 64
    provenance = build_initialization_provenance(
        model.state_dict(), pretrain_checkpoint_sha256=pretrain_sha256
    )
    config = {"project": {"initialization_provenance": provenance}}
    checkpoint = save_checkpoint(tmp_path / "来源 模型.ckpt", model, config=config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    extracted = extract_checkpoint_provenance(payload, checkpoint)

    assert extracted.checkpoint_sha256 == checkpoint_file_sha256(checkpoint)
    assert extracted.initialization_lineage == provenance["initialization_lineage"]
    assert extracted.pretrain_checkpoint_sha256 == pretrain_sha256
    assert extracted.parent_checkpoint_sha256 is None
    assert extracted.architecture_sha256 == state_dict_architecture_sha256(
        model.state_dict()
    )
    legacy = save_checkpoint(tmp_path / "legacy.ckpt", model, config={"project": {}})
    legacy_payload = torch.load(legacy, map_location="cpu", weights_only=False)
    with pytest.raises(SoupCompatibilityError, match="no initialization_provenance"):
        extract_checkpoint_provenance(legacy_payload, legacy)
    unsafe = extract_checkpoint_provenance(
        legacy_payload, legacy, require_complete=False
    )
    assert unsafe.initialization_lineage is None
    assert unsafe.provenance_source == "missing"


def test_checkpoint_provenance_rejects_architecture_tampering(tmp_path) -> None:
    model = torch.nn.Linear(2, 2)
    provenance = build_initialization_provenance(model.state_dict())
    provenance["architecture_sha256"] = "0" * 64
    checkpoint = save_checkpoint(
        tmp_path / "tampered.ckpt",
        model,
        config={"project": {"initialization_provenance": provenance}},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    with pytest.raises(SoupCompatibilityError, match="architecture"):
        extract_checkpoint_provenance(payload, checkpoint)


def test_initialization_provenance_rejects_two_starting_checkpoints() -> None:
    with pytest.raises(ValueError, match="parent and pretrain"):
        build_initialization_provenance(
            _state(0.0),
            parent_checkpoint_sha256="a" * 64,
            pretrain_checkpoint_sha256="b" * 64,
        )


def test_extract_checkpoint_state_handles_raw_ema_and_module_prefix() -> None:
    payload = {
        "model": {"module.weight": torch.tensor([1.0])},
        "ema": {"shadow": {"weight": torch.tensor([2.0])}, "num_updates": 5},
    }
    raw = extract_checkpoint_state(payload, weight_source="raw")
    ema = extract_checkpoint_state(payload, weight_source="ema")
    assert set(raw) == {"weight"}
    assert raw["weight"].item() == 1.0
    assert ema["weight"].item() == 2.0
    with pytest.raises(KeyError, match="swa"):
        extract_checkpoint_state(payload, weight_source="swa")


def test_strict_validation_contract_binds_full_jpg_roi_evidence(tmp_path: Path) -> None:
    manifest, audit, records = _validation_evidence(tmp_path)

    contract = build_soup_validation_contract(
        manifest,
        audit,
        records,
        target_marker="cd-68",
        evaluated_sample_count=2,
        metric_domain="jpg",
        audited_manifest_paths=[manifest],
    )
    payload = validate_soup_validation_contract(
        contract,
        require_strict=True,
        verify_files=True,
    )

    assert payload["safety_mode"] == "strict"
    assert payload["metric_domain"] == "jpg_roundtrip"
    assert payload["full_validation"] is True
    assert payload["filename_grid_verified"] is True
    assert payload["validation_sample_count"] == 2
    assert payload["per_image_record_count"] == 2
    assert len(payload["contract_sha256"]) == 64
    records.write_text(records.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SoupValidationContractError, match="Evidence file hash mismatch"):
        validate_soup_validation_contract(contract, verify_files=True)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"metric_domain": "float"}, "not_jpg_roundtrip"),
        ({"max_val_samples": 1}, "sample_limit"),
        ({"evaluated_sample_count": 1}, "sample_count_mismatch"),
    ],
)
def test_strict_validation_contract_rejects_nonpromotable_selection(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    manifest, audit, records = _validation_evidence(tmp_path)
    options: dict[str, object] = {
        "target_marker": "CD68",
        "evaluated_sample_count": 2,
    }
    options.update(kwargs)

    with pytest.raises(SoupValidationContractError, match=message):
        build_soup_validation_contract(manifest, audit, records, **options)


def test_unverified_roi_contract_requires_and_permanently_marks_unsafe_override(
    tmp_path: Path,
) -> None:
    manifest, audit, records = _validation_evidence(tmp_path)
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    audit_payload["filename_grid_verified"] = False
    audit_payload["context_enabled"] = False
    audit_payload["context_gate_reasons"] = ["unverified_filename_coordinates"]
    audit.write_text(json.dumps(audit_payload), encoding="utf-8")

    with pytest.raises(SoupValidationContractError, match="unverified_filename_roi_grid"):
        build_soup_validation_contract(
            manifest,
            audit,
            records,
            target_marker="CD68",
            evaluated_sample_count=2,
        )
    unsafe = build_soup_validation_contract(
        manifest,
        audit,
        records,
        target_marker="CD68",
        evaluated_sample_count=2,
        allow_unsafe_engineering_override=True,
    )
    assert unsafe.authoritative is False
    assert unsafe.unsafe_validation_override_used is True
    assert unsafe.unsafe_engineering_override_used is True
    assert "unverified_filename_roi_grid" in unsafe.unsafe_reasons
    with pytest.raises(SoupValidationContractError, match="Strict soup cannot use"):
        validate_soup_validation_contract(unsafe, require_strict=True)


def test_test_rows_cannot_enter_soup_even_with_unsafe_override(
    tmp_path: Path,
) -> None:
    manifest, audit, records = _validation_evidence(tmp_path)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["source_split"] = "official_test"
    _write_csv(manifest, rows)

    with pytest.raises(SoupValidationContractError, match="test data"):
        build_soup_validation_contract(
            manifest,
            audit,
            records,
            target_marker="CD68",
            evaluated_sample_count=2,
            allow_unsafe_engineering_override=True,
        )


def _soup_provenance(
    state: dict[str, torch.Tensor],
    contract: object,
    *,
    lineage: str,
) -> dict[str, object]:
    return bind_soup_validation_contract(
        {
            "common_initialization_lineage": lineage,
            "common_parent_checkpoint_sha256": None,
            "common_pretrain_checkpoint_sha256": None,
            "architecture_sha256": state_dict_architecture_sha256(state),
            "unsafe_lineage_override_used": False,
        },
        contract,  # type: ignore[arg-type]
    )


def test_model_soup_provenance_wins_over_training_initialization_and_rejects_unsafe(
    tmp_path: Path,
) -> None:
    manifest, audit, records = _validation_evidence(tmp_path)
    unsafe_contract = build_soup_validation_contract(
        manifest,
        audit,
        records,
        target_marker="CD68",
        evaluated_sample_count=1,
        max_val_samples=1,
        allow_unsafe_engineering_override=True,
    )
    model = torch.nn.Linear(2, 2)
    training_provenance = build_initialization_provenance(model.state_dict())
    soup_lineage = "c" * 64
    soup_provenance = _soup_provenance(
        model.state_dict(),
        unsafe_contract,
        lineage=soup_lineage,
    )
    checkpoint = save_checkpoint(
        tmp_path / "unsafe-soup-with-safe-training-provenance.ckpt",
        model,
        config={"project": {"initialization_provenance": training_provenance}},
        extra={"model_soup_provenance": soup_provenance},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    with pytest.raises(SoupCompatibilityError, match="cannot enter a strict soup"):
        extract_checkpoint_provenance(payload, checkpoint, require_complete=True)
    unsafe = extract_checkpoint_provenance(payload, checkpoint, require_complete=False)
    assert unsafe.provenance_source == "model_soup"
    assert unsafe.initialization_lineage == soup_lineage
    assert unsafe.initialization_lineage != training_provenance["initialization_lineage"]


@pytest.mark.parametrize(
    "flag",
    [
        "unsafe_lineage_override_used",
        "unsafe_validation_override_used",
        "unsafe_engineering_override_used",
    ],
)
def test_strict_reuse_rejects_every_unsafe_soup_flag(
    tmp_path: Path,
    flag: str,
) -> None:
    manifest, audit, records = _validation_evidence(tmp_path)
    contract = build_soup_validation_contract(
        manifest,
        audit,
        records,
        target_marker="CD68",
        evaluated_sample_count=2,
    )
    model = torch.nn.Linear(2, 2)
    lineage = build_initialization_provenance(model.state_dict())["initialization_lineage"]
    soup_provenance = _soup_provenance(
        model.state_dict(),
        contract,
        lineage=lineage,
    )
    soup_provenance[flag] = True
    checkpoint = save_checkpoint(
        tmp_path / f"{flag}.ckpt",
        model,
        extra={"model_soup_provenance": soup_provenance},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    with pytest.raises(SoupCompatibilityError, match="cannot enter a strict soup"):
        extract_checkpoint_provenance(payload, checkpoint)


def test_unsafe_soup_cannot_be_laundered_by_rebinding_strict_contract(
    tmp_path: Path,
) -> None:
    manifest, audit, records = _validation_evidence(tmp_path)
    strict_contract = build_soup_validation_contract(
        manifest,
        audit,
        records,
        target_marker="CD68",
        evaluated_sample_count=2,
    )
    unsafe_contract = build_soup_validation_contract(
        manifest,
        audit,
        records,
        target_marker="CD68",
        evaluated_sample_count=1,
        max_val_samples=1,
        allow_unsafe_engineering_override=True,
    )
    model = torch.nn.Linear(2, 2)
    lineage = build_initialization_provenance(model.state_dict())["initialization_lineage"]
    unsafe_provenance = _soup_provenance(
        model.state_dict(),
        unsafe_contract,
        lineage=lineage,
    )

    rebound = bind_soup_validation_contract(unsafe_provenance, strict_contract)
    assert rebound["safety_mode"] == "UNSAFE_ENGINEERING_OVERRIDE"
    assert rebound["unsafe_validation_override_used"] is True
    assert rebound["unsafe_engineering_override_used"] is True
    checkpoint = save_checkpoint(
        tmp_path / "rebound-still-unsafe.ckpt",
        model,
        extra={"model_soup_provenance": rebound},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    with pytest.raises(SoupCompatibilityError, match="cannot enter a strict soup"):
        extract_checkpoint_provenance(payload, checkpoint)


def test_safe_soup_validation_contract_allows_strict_reuse(tmp_path: Path) -> None:
    manifest, audit, records = _validation_evidence(tmp_path)
    contract = build_soup_validation_contract(
        manifest,
        audit,
        records,
        target_marker="CD68",
        evaluated_sample_count=2,
    )
    model = torch.nn.Linear(2, 2)
    lineage = build_initialization_provenance(model.state_dict())["initialization_lineage"]
    provenance = _soup_provenance(
        model.state_dict(),
        contract,
        lineage=lineage,
    )
    checkpoint = save_checkpoint(
        tmp_path / "strict-soup.ckpt",
        model,
        extra={"model_soup_provenance": provenance},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    extracted = extract_checkpoint_provenance(payload, checkpoint)

    assert extracted.provenance_source == "model_soup"
    assert extracted.initialization_lineage == lineage
