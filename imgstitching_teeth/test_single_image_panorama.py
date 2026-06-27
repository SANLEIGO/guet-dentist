from __future__ import annotations

import cv2
import numpy as np

from dental_stitcher_v1.pipeline import run_pipeline
from dental_stitcher_v1.segmentation import SegmentationResult


def test_run_pipeline_treats_single_image_as_panorama() -> None:
    image = np.zeros((160, 260, 3), dtype=np.uint8)
    image[48:112, 30:230] = (210, 220, 230)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (30, 48), (230, 112), 255, thickness=-1)
    overlay = image.copy()
    seg_result = SegmentationResult(mask=mask, overlay=overlay, method="test_mask")

    outputs = run_pipeline(
        [image],
        seg_results=[seg_result],
        enable_auto_calibration=False,
    )

    assert outputs.stitched is not None
    assert outputs.stitched_mask is not None
    assert outputs.diagnostics.quality_gate["single_image_panorama_mode"] is True
    assert outputs.diagnostics.quality_gate["accepted_indices"] == [0]
    assert outputs.diagnostics.registration.details["method"] == "identity"
    assert outputs.diagnostics.blending.details["method"] == "single_image_passthrough"

