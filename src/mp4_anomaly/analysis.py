"""Detection extraction and rolling statistics."""

from __future__ import annotations

import math
from collections import deque
from typing import Any

from .models import Detection, VideoMeta
from .settings import CLASS_ID_TO_NAME, CLASS_NAMES
from .video import format_time


class MovingCounter:
    """Maintain class counts over a fixed frame window."""

    def __init__(self, window_frames: int) -> None:
        self.window_frames = window_frames
        self.history: deque[dict[str, int]] = deque()
        self.sums = {name: 0 for name in CLASS_NAMES}

    def update(self, counts: dict[str, int]) -> bool:
        """Add one frame and return whether the window is full."""

        self.history.append(counts)

        for name in CLASS_NAMES:
            self.sums[name] += counts[name]

        if len(self.history) > self.window_frames:
            removed = self.history.popleft()
            for name in CLASS_NAMES:
                self.sums[name] -= removed[name]

        return len(self.history) == self.window_frames


def extract_detections(
    result: Any,
) -> tuple[dict[str, int], list[Detection], int]:
    """Extract finite and valid detections from one YOLO result."""

    counts = {name: 0 for name in CLASS_NAMES}
    detections: list[Detection] = []
    invalid_count = 0

    if result.boxes is None or len(result.boxes) == 0:
        return counts, detections, invalid_count

    height, width = result.orig_img.shape[:2]

    for box in result.boxes:
        class_value = float(box.cls[0].item())
        confidence = float(box.conf[0].item())
        coordinates = box.xyxy[0].detach().cpu().tolist()

        if len(coordinates) != 4:
            invalid_count += 1
            continue

        values = [class_value, confidence, *coordinates]
        if not all(math.isfinite(value) for value in values):
            invalid_count += 1
            continue

        class_id = int(class_value)
        if class_id not in CLASS_ID_TO_NAME:
            invalid_count += 1
            continue

        x1, y1, x2, y2 = map(round, coordinates)
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        if x2 <= x1 or y2 <= y1:
            invalid_count += 1
            continue

        class_name = CLASS_ID_TO_NAME[class_id]
        counts[class_name] += 1
        detections.append(
            Detection(
                class_name=class_name,
                confidence=confidence,
                box=(x1, y1, x2, y2),
            )
        )

    return counts, detections, invalid_count


def find_triggered_classes(
    moving_sums: dict[str, int],
    threshold: int,
    window_ready: bool,
) -> list[str]:
    """Return classes whose moving sum exceeds the threshold."""

    if not window_ready:
        return []

    return [
        name
        for name in CLASS_NAMES
        if moving_sums[name] > threshold
    ]


def build_stats_row(
    frame_number: int,
    counts: dict[str, int],
    moving_sums: dict[str, int],
    window_ready: bool,
    meta: VideoMeta,
    real_seconds_per_frame: float,
) -> dict[str, object]:
    """Build one frame statistics row."""

    real_seconds = frame_number * real_seconds_per_frame
    source_seconds = frame_number / meta.fps

    return {
        "frame_number": frame_number,
        "real_elapsed_seconds": real_seconds,
        "real_elapsed_time": format_time(real_seconds),
        "source_video_seconds": source_seconds,
        "source_video_time": format_time(source_seconds),
        "window_ready": window_ready,
        **counts,
        **{
            f"{name}_moving_sum": moving_sums[name]
            for name in CLASS_NAMES
        },
    }
