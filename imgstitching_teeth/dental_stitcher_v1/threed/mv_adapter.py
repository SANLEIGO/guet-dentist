from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .preprocess import Prepared3DAsset


@dataclass
class PseudoMultiviewPack:
    views_bgr: dict[str, np.ndarray]
    views_bgra: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def preview_grid(self) -> np.ndarray:
        ordered_tags = [tag for tag in ("front", "left", "right", "back") if tag in self.views_bgr]
        tiles: list[np.ndarray] = []
        for tag in ordered_tags:
            tile = self.views_bgr[tag].copy()
            cv2.putText(
                tile,
                tag.upper(),
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (30, 30, 30),
                2,
                cv2.LINE_AA,
            )
            tiles.append(tile)

        if not tiles:
            raise ValueError("No pseudo multiview images available for preview.")

        if len(tiles) == 1:
            return tiles[0]
        if len(tiles) == 2:
            return np.concatenate(tiles, axis=1)

        if len(tiles) == 3:
            blank = np.full_like(tiles[0], 255)
            cv2.putText(blank, "RESERVED", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
            top = np.concatenate(tiles[:2], axis=1)
            bottom = np.concatenate([tiles[2], blank], axis=1)
            return np.concatenate([top, bottom], axis=0)

        top = np.concatenate(tiles[:2], axis=1)
        bottom = np.concatenate(tiles[2:4], axis=1)
        return np.concatenate([top, bottom], axis=0)

    def archive_bytes(self, transparent: bool = False) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            source = self.views_bgra if transparent else self.views_bgr
            for tag, image in source.items():
                success, encoded = cv2.imencode(".png", image)
                if not success:
                    raise ValueError(f"Failed to encode pseudo view {tag}.")
                suffix = "_rgba" if transparent else ""
                zf.writestr(f"{tag}{suffix}.png", encoded.tobytes())
            zf.writestr(
                "metadata.json",
                json.dumps(self.metadata, ensure_ascii=False, indent=2).encode("utf-8"),
            )
        buffer.seek(0)
        return buffer.getvalue()

    def payload_images(self, transparent: bool = True) -> dict[str, np.ndarray]:
        return self.views_bgra if transparent else self.views_bgr


def build_pseudo_multiview_pack(
    prepared_asset: Prepared3DAsset,
    include_back: bool = False,
    side_strength: float = 0.10,
) -> PseudoMultiviewPack:
    front_rgba = prepared_asset.bgra_image.copy()
    front_bgr = prepared_asset.bgr_image.copy()

    views_bgra = {
        "front": front_rgba,
        "left": _pseudo_view(front_rgba, "left", side_strength),
        "right": _pseudo_view(front_rgba, "right", side_strength),
    }

    if include_back:
        views_bgra["back"] = _pseudo_view(front_rgba, "back", side_strength)

    views_bgr = {tag: _rgba_to_white_bgr(image) for tag, image in views_bgra.items()}
    views_bgr["front"] = front_bgr

    metadata = {
        "method": "pseudo_multiview_from_single_arch",
        "views": list(views_bgr.keys()),
        "side_strength": float(side_strength),
        "include_back": bool(include_back),
        "source_preprocess": prepared_asset.metadata,
        "notes": [
            "These are synthetic support views derived from a single stitched dental arch image.",
            "They are suitable for demo conditioning but should not be interpreted as clinically accurate hidden geometry.",
        ],
    }
    return PseudoMultiviewPack(views_bgr=views_bgr, views_bgra=views_bgra, metadata=metadata)


def _pseudo_view(rgba_image: np.ndarray, direction: str, side_strength: float) -> np.ndarray:
    h, w = rgba_image.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dx = w * side_strength
    dy = h * side_strength * 0.12

    if direction == "left":
        dst = np.float32([
            [dx, dy],
            [w - dx * 0.55, 0],
            [w - dx * 0.55, h - 1],
            [dx, h - 1 - dy],
        ])
    elif direction == "right":
        dst = np.float32([
            [0, 0],
            [w - 1 - dx, dy],
            [w - 1 - dx, h - 1 - dy],
            [dx * 0.55, h - 1],
        ])
    elif direction == "back":
        dst = np.float32([
            [dx * 0.5, dy],
            [w - 1 - dx * 0.5, dy],
            [w - 1 - dx * 0.5, h - 1 - dy],
            [dx * 0.5, h - 1 - dy],
        ])
    else:
        raise ValueError(f"Unsupported pseudo view direction: {direction}")

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        rgba_image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    if direction == "back":
        warped = cv2.flip(warped, 1)
        warped = _apply_directional_shading(warped, direction="back", amplitude=0.05)
    else:
        warped = _apply_directional_shading(warped, direction=direction, amplitude=0.04)

    return warped


def _apply_directional_shading(rgba_image: np.ndarray, direction: str, amplitude: float) -> np.ndarray:
    h, w = rgba_image.shape[:2]
    ramp = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    if direction == "left":
        gain = 1.0 + amplitude * (-ramp)
    elif direction == "right":
        gain = 1.0 + amplitude * ramp
    else:
        gain = 1.0 - amplitude * np.abs(ramp)

    gain = gain.reshape(1, w, 1)
    result = rgba_image.copy().astype(np.float32)
    result[..., :3] *= gain
    result[..., :3] = np.clip(result[..., :3], 0, 255)
    return result.astype(np.uint8)


def _rgba_to_white_bgr(rgba_image: np.ndarray) -> np.ndarray:
    alpha = rgba_image[..., 3:4].astype(np.float32) / 255.0
    foreground = rgba_image[..., :3].astype(np.float32)
    background = np.full_like(foreground, 255.0)
    blended = foreground * alpha + background * (1.0 - alpha)
    return np.clip(blended, 0, 255).astype(np.uint8)
