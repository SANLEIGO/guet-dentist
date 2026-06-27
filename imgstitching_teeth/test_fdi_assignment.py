from __future__ import annotations

from dental_stitcher_v1.template3d import ArchLabel
from dental_stitcher_v1.template3d.fdi_assignment import (
    FDISequenceConfig,
    assign_tooth_candidates_sequentially,
    build_expected_fdi_sequence,
)
from test_instance_extractor import _make_instance


def test_build_expected_fdi_sequence_filters_missing_and_wisdom() -> None:
    config = FDISequenceConfig(
        arch_label=ArchLabel.LOWER,
        image_left_is_patient_right=True,
        include_patient_right_wisdom=False,
        include_patient_left_wisdom=True,
        missing_tooth_ids=[46, 41],
    )

    sequence = build_expected_fdi_sequence(config)

    assert sequence[:5] == [47, 45, 44, 43, 42]
    assert 46 not in sequence
    assert 41 not in sequence
    assert 48 not in sequence
    assert sequence[-1] == 38


def test_assign_tooth_candidates_sequentially_marks_mismatch_for_review() -> None:
    instances = [
        _make_instance(instance_id=0, bbox=(10, 10, 20, 20)),
        _make_instance(instance_id=1, bbox=(25, 10, 35, 20)),
        _make_instance(instance_id=2, bbox=(40, 10, 50, 20)),
    ]
    config = FDISequenceConfig(
        arch_label=ArchLabel.UPPER,
        image_left_is_patient_right=True,
        include_patient_right_wisdom=False,
        include_patient_left_wisdom=False,
    )

    result = assign_tooth_candidates_sequentially(instances, config, anchor="left")

    assert result.review_required is True
    assert [item.tooth_id for item in result.assignments] == [17, 16, 15]
    assert all(item.review_required for item in result.assignments)
