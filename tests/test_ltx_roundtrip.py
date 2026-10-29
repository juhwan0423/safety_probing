from __future__ import annotations

import pytest
import torch

from safety_probing.ltx_roundtrip import (
    _part_complete,
    decoded_chunks_to_encoder_input,
    select_condition_rows,
)
from safety_probing.ltx_roundtrip_evaluate import (
    evaluate_control_gate,
    pathway_gap_persistence,
    roundtrip_layout,
)


def test_decoded_chunks_convert_to_encoder_range() -> None:
    first = torch.zeros(2, 4, 6, 3)
    second = torch.ones(3, 4, 6, 3)
    video, audit = decoded_chunks_to_encoder_input(
        [first, second], expected_frames=5
    )
    assert video.shape == (1, 3, 5, 4, 6)
    assert float(video[:, :, :2].max()) == -1.0
    assert float(video[:, :, 2:].min()) == 1.0
    assert audit["decoded_frame_count"] == 5


def test_decoded_chunks_reject_bad_frame_count() -> None:
    with pytest.raises(RuntimeError, match="Expected 5 decoded frames"):
        decoded_chunks_to_encoder_input(
            [torch.zeros(4, 4, 6, 3)], expected_frames=5
        )


def test_select_condition_rows_preserves_explicit_order() -> None:
    rows = [
        {"condition_id": "b", "splits": {"main": "test"}},
        {"condition_id": "a", "splits": {"main": "val"}},
        {"condition_id": "c", "splits": {"main": "train"}},
    ]
    selected = select_condition_rows(
        rows,
        splits={"val", "test"},
        condition_ids=["a", "b"],
    )
    assert [row["condition_id"] for row in selected] == ["a", "b"]


def test_part_complete_checks_contract_shape_and_finiteness(tmp_path) -> None:
    path = tmp_path / "part.pt"
    torch.save(
        {
            "fingerprint": "ok",
            "feature": torch.zeros(7, 10, 18, 128, dtype=torch.float16),
        },
        path,
    )
    assert _part_complete(path, fingerprint="ok")
    assert not _part_complete(path, fingerprint="changed")


def test_roundtrip_layout_groups_identical_seed_coverage() -> None:
    target_records = [
        {"condition_id": "a"},
        {"condition_id": "b"},
    ]
    cache = {
        "records": [
            {"condition_id": "a", "decoder_seed": 17},
            {"condition_id": "a", "decoder_seed": 29},
            {"condition_id": "b", "decoder_seed": 17},
            {"condition_id": "b", "decoder_seed": 29},
        ]
    }
    seeds, by_seed, indices, records = roundtrip_layout(cache, target_records)
    assert seeds == [17, 29]
    assert by_seed == {17: [0, 2], 29: [1, 3]}
    assert indices == [0, 1]
    assert records == target_records


def _aggregate(values: dict[str, float]) -> dict[str, dict[str, float]]:
    return {key: {"mean": value} for key, value in values.items()}


def test_control_gate_passes_small_roundtrip_drift() -> None:
    native = _aggregate(
        {
            "splits.test.future_tip.auroc": 1.0,
            "splits.test.static.load_mass_kg.mae": 10.0,
            "splits.test.static.load_com_lateral_m.mae": 0.03,
            "splits.test.future_state.roll_deg.mae": 2.0,
            "splits.test.future_summary.min_zmp_outward_margin_m.mae": 0.03,
            "pairs.test.hidden_property_delta.load_mass.direction_accuracy": 1.0,
            "pairs.test.hidden_property_delta.com_shift.direction_accuracy": 0.99,
        }
    )
    roundtrip = _aggregate(
        {
            "splits.test.future_tip.auroc": 0.99,
            "splits.test.static.load_mass_kg.mae": 11.0,
            "splits.test.static.load_com_lateral_m.mae": 0.032,
            "splits.test.future_state.roll_deg.mae": 2.2,
            "splits.test.future_summary.min_zmp_outward_margin_m.mae": 0.034,
            "pairs.test.hidden_property_delta.load_mass.direction_accuracy": 0.98,
            "pairs.test.hidden_property_delta.com_shift.direction_accuracy": 0.96,
        }
    )
    result = evaluate_control_gate(
        native,
        roundtrip,
        {
            "max_tip_auroc_drop": 0.02,
            "max_primary_mae_ratio": 1.25,
            "max_intervention_direction_accuracy_drop": 0.05,
        },
    )
    assert result["passed"] is True


def test_control_gate_fails_large_gt_degradation() -> None:
    native = _aggregate(
        {
            "splits.test.future_tip.auroc": 1.0,
            "splits.test.static.load_mass_kg.mae": 10.0,
            "splits.test.static.load_com_lateral_m.mae": 0.03,
            "splits.test.future_state.roll_deg.mae": 2.0,
            "splits.test.future_summary.min_zmp_outward_margin_m.mae": 0.03,
            "pairs.test.hidden_property_delta.load_mass.direction_accuracy": 1.0,
            "pairs.test.hidden_property_delta.com_shift.direction_accuracy": 0.99,
        }
    )
    roundtrip = _aggregate(
        {
            "splits.test.future_tip.auroc": 0.90,
            "splits.test.static.load_mass_kg.mae": 20.0,
            "splits.test.static.load_com_lateral_m.mae": 0.06,
            "splits.test.future_state.roll_deg.mae": 4.0,
            "splits.test.future_summary.min_zmp_outward_margin_m.mae": 0.06,
            "pairs.test.hidden_property_delta.load_mass.direction_accuracy": 0.80,
            "pairs.test.hidden_property_delta.com_shift.direction_accuracy": 0.70,
        }
    )
    result = evaluate_control_gate(
        native,
        roundtrip,
        {
            "max_tip_auroc_drop": 0.02,
            "max_primary_mae_ratio": 1.25,
            "max_intervention_direction_accuracy_drop": 0.05,
        },
    )
    assert result["passed"] is False


def test_pathway_gap_persistence_separates_roundtrip_effect() -> None:
    path = "splits.test.static.load_mass_kg.mae"
    aggregate = {
        "gt_native": {path: {"mean": 10.0}},
        "gt_roundtrip": {path: {"mean": 12.0}},
        "generated_native": {path: {"mean": 90.0}},
        "generated_roundtrip": {path: {"mean": 92.0}},
    }
    # Populate the remaining required metrics with the same synthetic values.
    for metric_path in (
        "splits.test.static.load_com_lateral_m.mae",
        "splits.test.future_state.roll_deg.mae",
        "splits.test.future_summary.min_zmp_outward_margin_m.mae",
    ):
        aggregate["gt_native"][metric_path] = {"mean": 10.0}
        aggregate["gt_roundtrip"][metric_path] = {"mean": 12.0}
        aggregate["generated_native"][metric_path] = {"mean": 90.0}
        aggregate["generated_roundtrip"][metric_path] = {"mean": 92.0}
    result = pathway_gap_persistence(aggregate)
    assert result["load_mass_kg_mae"]["gap_persistence_ratio"] == 1.0
    assert result["load_mass_kg_mae"]["gt_roundtrip_minus_native"] == 2.0
