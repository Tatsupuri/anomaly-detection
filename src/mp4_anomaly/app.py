"""Application orchestration."""

from __future__ import annotations

import math
from collections import deque

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from .analysis import (
    MovingCounter,
    build_stats_row,
    extract_detections,
    find_triggered_classes,
)
from .events import event_to_row, start_event, update_event
from .exporters import plot_moving_sums, write_csv
from .models import ActiveEvent, AppConfig, FramePacket
from .settings import ANOMALY_FIELDS, CLASS_IDS
from .video import read_video_meta, write_packet


def run(config: AppConfig) -> None:
    """Run analysis and export all outputs."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = config.output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    meta = read_video_meta(config.input_path)

    window_frames = max(
        1,
        math.ceil(
            config.window_seconds
            / config.real_seconds_per_frame
        ),
    )
    pre_frames = max(
        0,
        math.ceil(
            config.pre_seconds
            / config.real_seconds_per_frame
        ),
    )
    post_frames = max(
        0,
        math.ceil(
            config.post_seconds
            / config.real_seconds_per_frame
        ),
    )

    model = YOLO(config.model)
    results = model.predict(
        source=str(config.input_path),
        stream=True,
        classes=CLASS_IDS,
        conf=config.confidence,
        imgsz=config.image_size,
        device=config.device,
        batch=config.batch_size,
        half=config.use_half,
        vid_stride=1,
        verbose=False,
    )

    moving_counter = MovingCounter(window_frames)
    pre_buffer: deque[FramePacket] = deque(
        maxlen=pre_frames + 1
    )

    stats_rows: list[dict[str, object]] = []
    anomaly_rows: list[dict[str, object]] = []

    writer: cv2.VideoWriter | None = None
    event: ActiveEvent | None = None

    event_id = 0
    post_remaining = 0
    last_written_frame = -1
    invalid_detections = 0

    progress = tqdm(
        enumerate(results),
        total=meta.frame_count or None,
        unit="frame",
        desc="Analyzing",
        dynamic_ncols=True,
    )

    try:
        for frame_number, result in progress:
            counts, detections, invalid_count = extract_detections(
                result
            )
            invalid_detections += invalid_count

            window_ready = moving_counter.update(counts)
            triggered = find_triggered_classes(
                moving_counter.sums,
                config.threshold,
                window_ready,
            )
            is_anomaly = bool(triggered)

            packet = FramePacket(
                frame_number=frame_number,
                image=result.orig_img.copy(),
                detections=detections,
                moving_sums=dict(moving_counter.sums),
            )
            pre_buffer.append(packet)

            stats_rows.append(
                build_stats_row(
                    frame_number,
                    counts,
                    moving_counter.sums,
                    window_ready,
                    meta,
                    config.real_seconds_per_frame,
                )
            )

            if writer is None:
                if is_anomaly:
                    event_id += 1
                    writer, event, last_written_frame = start_event(
                        event_id,
                        clips_dir,
                        pre_buffer,
                        triggered,
                        moving_counter.sums,
                        meta,
                        config,
                    )
                    post_remaining = post_frames
                continue

            if event is None:
                raise RuntimeError("Missing active event state")

            if is_anomaly:
                if frame_number > last_written_frame:
                    write_packet(
                        writer,
                        packet,
                        config.real_seconds_per_frame,
                        meta.fps,
                        config.review_fps,
                    )
                    last_written_frame = frame_number

                update_event(
                    event,
                    triggered,
                    moving_counter.sums,
                    frame_number,
                )
                post_remaining = post_frames
                continue

            if post_remaining > 0:
                if frame_number > last_written_frame:
                    write_packet(
                        writer,
                        packet,
                        config.real_seconds_per_frame,
                        meta.fps,
                        config.review_fps,
                    )
                    last_written_frame = frame_number

                event.clip_end_frame = frame_number
                post_remaining -= 1

                if post_remaining > 0:
                    continue

            writer.release()
            writer = None
            anomaly_rows.append(
                event_to_row(
                    event,
                    meta,
                    config.real_seconds_per_frame,
                )
            )
            event = None

    except Exception:
        if writer is not None:
            writer.release()
        raise

    if writer is not None and event is not None:
        writer.release()
        anomaly_rows.append(
            event_to_row(
                event,
                meta,
                config.real_seconds_per_frame,
            )
        )

    write_csv(
        config.output_dir / "moving_sum.csv",
        stats_rows,
    )
    write_csv(
        config.output_dir / "anomalies.csv",
        anomaly_rows,
        ANOMALY_FIELDS,
    )
    plot_moving_sums(
        config.output_dir / "moving_sum.png",
        stats_rows,
        anomaly_rows,
        config.threshold,
    )

    tqdm.write(
        (
            f"Done: {len(anomaly_rows)} anomalies, "
            f"{invalid_detections} invalid detections skipped -> "
            f"{config.output_dir.resolve()}"
        )
    )
