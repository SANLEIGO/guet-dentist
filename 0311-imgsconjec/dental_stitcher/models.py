from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ImageRecord:
    path: Path
    arch: str
    segment: str
    image: np.ndarray
    display_name: str = field(init=False)
    quality_score: float = 0.0
    sharpness_score: float = 0.0
    exposure_score: float = 0.0

    def __post_init__(self) -> None:
        self.display_name = self.path.name


@dataclass
class MatchResult:
    success: bool
    score: float
    inliers: int
    homography: np.ndarray | None
    inverse_homography: np.ndarray | None
    details: dict[str, Any] = field(default_factory=dict)
    sequence_distance: int = 0
    weighted_score: float = 0.0
    matched_points0: np.ndarray | None = None
    matched_points1: np.ndarray | None = None


@dataclass
class StitchResult:
    success: bool
    anchor_index: int | None
    panorama: np.ndarray | None
    logs: list[str]
    method_name: str
    included_indices: list[int] = field(default_factory=list)
    ordered_indices: list[int] = field(default_factory=list)
    pairwise_matches: dict[tuple[int, int], MatchResult] = field(default_factory=dict)


@dataclass
class CandidateScore:
    index: int
    display_name: str
    quality_score: float
    connectivity_score: float
    partner_count: int
    total_score: float
    recommended: bool = False


@dataclass
class PrecheckItem:
    index: int
    display_name: str
    sharpness_score: float
    exposure_score: float
    quality_score: float
    keep: bool
    reason: str


@dataclass
class PrecheckReport:
    items: list[PrecheckItem]
    kept_indices: list[int]
    dropped_indices: list[int]
