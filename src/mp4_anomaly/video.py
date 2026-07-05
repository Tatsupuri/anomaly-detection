"""Video metadata, annotation, and clip writing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from .models import FramePacket, VideoMeta
from .settings import CLASS_NAMES


def read_video_meta(path: Path) -> VideoMeta:
    """Read source video metadata."""

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    try:
        meta = VideoMeta(
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()

    if meta.fps <= 0:
        raise RuntimeError("Cannot read source FPS")
    if meta.width <= 0 or meta.height <= 0:
        raise RuntimeError("Cannot read source resolution")

    return meta


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""

    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def annotate_frame(
    packet: FramePacket,
    real_seconds_per_frame: float,
    source_fps: float,
) -> Any:
    """Draw detections, timestamps, and moving sums."""

    image = packet.image.copy()

    for detection in packet.detections:
        x1, y1, x2, y2 = detection.box

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            image,
            f"{detection.class_name} {detection.confidence:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    real_time = format_time(
        packet.frame_number * real_seconds_per_frame
    )
    source_time = format_time(packet.frame_number / source_fps)
    sums = " ".join(
        f"{name}={packet.moving_sums[name]}"
        for name in CLASS_NAMES
    )

    lines = [
        f"real: {real_time}  source: {source_time}",
        f"moving sum: {sums}",
    ]

    for line_number, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (12, 28 + line_number * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return image


def open_clip_writer(
    path: Path,
    meta: VideoMeta,
    review_fps: float,
) -> cv2.VideoWriter:
    """Open an MP4 writer."""

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        review_fps,
        (meta.width, meta.height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot create clip: {path}")

    return writer


def write_packet(
    writer: cv2.VideoWriter,
    packet: FramePacket,
    real_seconds_per_frame: float,
    source_fps: float,
    review_fps: float,
) -> None:
    """Write one analyzed frame at review speed."""

    image = annotate_frame(
        packet,
        real_seconds_per_frame,
        source_fps,
    )
    repeat = max(1, round(review_fps * real_seconds_per_frame))

    for _ in range(repeat):
        writer.write(image)
