"""Shared constants."""

CLASS_ID_TO_NAME = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CLASS_IDS = list(CLASS_ID_TO_NAME)
CLASS_NAMES = list(CLASS_ID_TO_NAME.values())

ANOMALY_FIELDS = [
    "event_id",
    "trigger_classes",
    "peak_class",
    "peak_moving_sum",
    "clip_start_frame",
    "clip_end_frame",
    "threshold_start_frame",
    "threshold_end_frame",
    "clip_start_real_seconds",
    "clip_start_real_time",
    "clip_end_real_seconds",
    "clip_end_real_time",
    "threshold_start_real_seconds",
    "threshold_start_real_time",
    "threshold_end_real_seconds",
    "threshold_end_real_time",
    "clip_start_source_seconds",
    "clip_start_source_time",
    "clip_end_source_seconds",
    "clip_end_source_time",
    "threshold_start_source_seconds",
    "threshold_start_source_time",
    "threshold_end_source_seconds",
    "threshold_end_source_time",
    "clip_path",
]
