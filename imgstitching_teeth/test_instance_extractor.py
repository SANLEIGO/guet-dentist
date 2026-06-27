from __future__ import annotations

import numpy as np

from dental_stitcher_v1.calibration.instance_extractor import (
    ToothInstance,
    _normalize_instance_mask,
    merge_overlapping_instances,
)


def _make_instance(
    *,
    instance_id: int,
    bbox: tuple[int, int, int, int],
    area: int | None = None,
    confidence: float = 0.5,
) -> ToothInstance:
    x1, y1, x2, y2 = bbox
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    area_value = int(np.count_nonzero(mask)) if area is None else area
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    return ToothInstance(
        instance_id=instance_id,
        class_id=0,
        class_name="Abrasion",
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        center=center,
        mask=mask,
        area=area_value,
        aspect_ratio=(x2 - x1) / max(y2 - y1, 1),
        confidence=confidence,
        width=x2 - x1,
        height=y2 - y1,
        source_instance_ids=[instance_id],
        source_labels=["Abrasion"],
    )


def test_normalize_instance_mask_resizes_to_image_shape() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:10, 5:20] = 255

    normalized = _normalize_instance_mask(mask, (40, 60))

    assert normalized.shape == (40, 60)
    assert normalized.dtype == np.uint8
    assert int(np.count_nonzero(normalized)) > 0


def test_merge_overlapping_instances_combines_same_tooth_masks() -> None:
    left = _make_instance(instance_id=0, bbox=(10, 10, 34, 40), confidence=0.91)
    overlap = _make_instance(instance_id=1, bbox=(15, 12, 31, 37), confidence=0.62)
    right = _make_instance(instance_id=2, bbox=(60, 12, 84, 40), confidence=0.88)

    merged = merge_overlapping_instances([left, overlap, right], overlap_threshold=0.5)

    assert len(merged) == 2
    assert merged[0].source_instance_ids == [0, 1]
    assert merged[1].source_instance_ids == [2]
