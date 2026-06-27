from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from dental_stitcher_v1.calibration.instance_extractor import ToothInstance, sort_instances_by_position
from dental_stitcher_v1.template3d.schema import ArchLabel


AssignmentAnchor = Literal["left", "right"]


@dataclass(frozen=True)
class FDISequenceConfig:
    arch_label: ArchLabel
    image_left_is_patient_right: bool = True
    include_patient_right_wisdom: bool = True
    include_patient_left_wisdom: bool = True
    missing_tooth_ids: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class AssignedToothCandidate:
    tooth_id: int
    instance: ToothInstance
    review_required: bool = False
    notes: str = ""


@dataclass(frozen=True)
class FDIAssignmentResult:
    assignments: list[AssignedToothCandidate]
    expected_sequence: list[int]
    review_required: bool
    notes: list[str]


def build_expected_fdi_sequence(config: FDISequenceConfig) -> list[int]:
    base_sequence = _base_fdi_sequence(config.arch_label)
    if not config.image_left_is_patient_right:
        base_sequence = list(reversed(base_sequence))

    filtered: list[int] = []
    missing = set(config.missing_tooth_ids)
    for tooth_id in base_sequence:
        if tooth_id in missing:
            continue
        if tooth_id % 10 == 8:
            if _is_patient_right_tooth(tooth_id) and not config.include_patient_right_wisdom:
                continue
            if _is_patient_left_tooth(tooth_id) and not config.include_patient_left_wisdom:
                continue
        filtered.append(tooth_id)
    return filtered


def assign_tooth_candidates_sequentially(
    instances: list[ToothInstance],
    config: FDISequenceConfig,
    *,
    anchor: AssignmentAnchor = "left",
) -> FDIAssignmentResult:
    ordered_instances = sort_instances_by_position(instances)
    expected_sequence = build_expected_fdi_sequence(config)
    notes: list[str] = []
    review_required = False

    if not ordered_instances:
        return FDIAssignmentResult(
            assignments=[],
            expected_sequence=expected_sequence,
            review_required=True,
            notes=["没有可编号的牙齿候选实例。"],
        )

    if not expected_sequence:
        return FDIAssignmentResult(
            assignments=[],
            expected_sequence=[],
            review_required=True,
            notes=["目标 FDI 序列为空，请检查牙弓、缺牙和智齿配置。"],
        )

    instance_count = len(ordered_instances)
    sequence_count = len(expected_sequence)
    if instance_count != sequence_count:
        review_required = True
        notes.append(
            f"实例数量 {instance_count} 与目标牙位数量 {sequence_count} 不一致，"
            f"已按{ '左侧' if anchor == 'left' else '右侧' }起始顺序做候选编号，需人工复核。"
        )

    if anchor == "right":
        target_sequence = expected_sequence[-instance_count:]
    else:
        target_sequence = expected_sequence[:instance_count]

    assignments: list[AssignedToothCandidate] = []
    for instance, tooth_id in zip(ordered_instances, target_sequence):
        assignments.append(
            AssignedToothCandidate(
                tooth_id=tooth_id,
                instance=instance,
                review_required=review_required,
                notes=notes[0] if review_required else "",
            )
        )

    if instance_count > sequence_count:
        notes.append("检测到的候选实例多于人工设定的牙位数量，超出的候选未参与编号。")

    return FDIAssignmentResult(
        assignments=assignments,
        expected_sequence=expected_sequence,
        review_required=review_required,
        notes=notes,
    )


def _base_fdi_sequence(arch_label: ArchLabel) -> list[int]:
    if arch_label == ArchLabel.UPPER:
        return [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
    if arch_label == ArchLabel.LOWER:
        return [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]
    return []


def _is_patient_right_tooth(tooth_id: int) -> bool:
    return tooth_id // 10 in {1, 4, 5, 8}


def _is_patient_left_tooth(tooth_id: int) -> bool:
    return tooth_id // 10 in {2, 3, 6, 7}
