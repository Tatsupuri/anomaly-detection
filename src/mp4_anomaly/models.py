"""Application data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppConfig:
    """Store validated application settings."""

    input_path: Path
    output_dir: Path
    model: str
    confidence: float
    image_size: int
    device: str
    batch_size: int
    use_half: bool
    real_seconds_per_frame: float
    window_seconds: float
    threshold: int
    pre_seconds: float
    post_seconds: float
    review_fps: float


@dataclass(frozen=True)
class VideoMeta:
    """Store source video metadata."""

    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class Detection:
    """Store one valid detection."""

    class_name: str
    confidence: float
    box: tuple[int, int, int, int]


@dataclass
class FramePacket:
    """Store one analyzed frame."""

    frame_number: int
    image: Any
    detections: list[Detection]
    moving_sums: dict[str, int]


@dataclass
class ActiveEvent:
    """Track one active anomaly event."""

    event_id: int
    clip_path: Path
    clip_start_frame: int
    threshold_start_frame: int
    last_anomaly_frame: int
    clip_end_frame: int
    trigger_classes: set[str] = field(default_factory=set)
    peak_class: str = ""
    peak_moving_sum: int = 0
