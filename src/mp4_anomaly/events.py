"""Anomaly event lifecycle helpers."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import cv2

from .models import ActiveEvent, AppConfig, FramePacket, VideoMeta
from .settings import CLASS_NAMES
from .video import format_time, open_clip_writer, write_packet


def start_event(
    event_id: int,
    clips_dir: Path,
    pre_buffer: deque[FramePacket],
    triggered: list[str],
    moving_sums: dict[str, int],
    meta: VideoMeta,
    config: AppConfig,
) -> tuple[cv2.VideoWriter, ActiveEvent, int]:
    """Start one anomaly clip."""

    current_packet = pre_buffer[-1]
    clip_path = clips_dir / f"anomaly_{event_id:04d}.mp4"
    writer = open_clip_writer(
        clip_path,
        meta,
        config.review_fps,
    )

    for packet in pre_buffer:
        write_packet(
            writer,
            packet,
            config.real_seconds_per_frame,
            meta.fps,
            config.review_fps,
        )

    peak_class = max(
        CLASS_NAMES,
        key=lambda name: moving_sums[name],
    )

    event = ActiveEvent(
        event_id=event_id,
        clip_path=clip_path,
        clip_start_frame=pre_buffer[0].frame_number,
        threshold_start_frame=current_packet.frame_number,
        last_anomaly_frame=current_packet.frame_number,
        clip_end_frame=current_packet.frame_number,
        trigger_classes=set(triggered),
        peak_class=peak_class,
        peak_moving_sum=moving_sums[peak_class],
    )

    return writer, event, current_packet.frame_number


def update_event(
    event: ActiveEvent,
    triggered: list[str],
    moving_sums: dict[str, int],
    frame_number: int,
) -> None:
    """Update one active anomaly event."""

    event.last_anomaly_frame = frame_number
    event.clip_end_frame = frame_number
    event.trigger_classes.update(triggered)

    peak_class = max(
        CLASS_NAMES,
        key=lambda name: moving_sums[name],
    )
    peak_value = moving_sums[peak_class]

    if peak_value > event.peak_moving_sum:
        event.peak_class = peak_class
        event.peak_moving_sum = peak_value


def event_to_row(
    event: ActiveEvent,
    meta: VideoMeta,
    real_seconds_per_frame: float,
) -> dict[str, object]:
    """Convert one event to a CSV row."""

    def real_seconds(frame_number: int) -> float:
        return frame_number * real_seconds_per_frame

    def source_seconds(frame_number: int) -> float:
        return frame_number / meta.fps

    clip_start_real = real_seconds(event.clip_start_frame)
    clip_end_real = real_seconds(event.clip_end_frame)
    threshold_start_real = real_seconds(event.threshold_start_frame)
    threshold_end_real = real_seconds(event.last_anomaly_frame)

    clip_start_source = source_seconds(event.clip_start_frame)
    clip_end_source = source_seconds(event.clip_end_frame)
    threshold_start_source = source_seconds(event.threshold_start_frame)
    threshold_end_source = source_seconds(event.last_anomaly_frame)

    return {
        "event_id": event.event_id,
        "trigger_classes": ",".join(sorted(event.trigger_classes)),
        "peak_class": event.peak_class,
        "peak_moving_sum": event.peak_moving_sum,
        "clip_start_frame": event.clip_start_frame,
        "clip_end_frame": event.clip_end_frame,
        "threshold_start_frame": event.threshold_start_frame,
        "threshold_end_frame": event.last_anomaly_frame,
        "clip_start_real_seconds": clip_start_real,
        "clip_start_real_time": format_time(clip_start_real),
        "clip_end_real_seconds": clip_end_real,
        "clip_end_real_time": format_time(clip_end_real),
        "threshold_start_real_seconds": threshold_start_real,
        "threshold_start_real_time": format_time(threshold_start_real),
        "threshold_end_real_seconds": threshold_end_real,
        "threshold_end_real_time": format_time(threshold_end_real),
        "clip_start_source_seconds": clip_start_source,
        "clip_start_source_time": format_time(clip_start_source),
        "clip_end_source_seconds": clip_end_source,
        "clip_end_source_time": format_time(clip_end_source),
        "threshold_start_source_seconds": threshold_start_source,
        "threshold_start_source_time": format_time(
            threshold_start_source
        ),
        "threshold_end_source_seconds": threshold_end_source,
        "threshold_end_source_time": format_time(threshold_end_source),
        "clip_path": str(event.clip_path),
    }
