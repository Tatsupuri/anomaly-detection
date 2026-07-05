"""Command-line argument handling."""

from __future__ import annotations

import argparse
from pathlib import Path

from .models import AppConfig


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Detect moving-sum anomalies in a time-lapse MP4 and "
            "export annotated review clips."
        )
    )

    parser.add_argument("input", type=Path, help="Input MP4 file")
    parser.add_argument("output", type=Path, help="Output directory")
    parser.add_argument("--model", default="yolo11m.pt", help="YOLO model")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device or cpu",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Inference batch size",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 inference",
    )
    parser.add_argument(
        "--real-seconds-per-frame",
        type=float,
        default=2.0,
        help="Real seconds represented by one source frame",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=60.0,
        help="Moving-sum window in real seconds",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="Anomaly threshold; the moving sum must exceed it",
    )
    parser.add_argument(
        "--pre-seconds",
        type=float,
        default=5.0,
        help="Real seconds to include before an anomaly",
    )
    parser.add_argument(
        "--post-seconds",
        type=float,
        default=5.0,
        help="Real seconds to include after an anomaly",
    )
    parser.add_argument(
        "--review-fps",
        type=float,
        default=10.0,
        help="Output review clip FPS",
    )

    return parser


def parse_config() -> AppConfig:
    """Parse and validate command-line arguments."""

    args = build_parser().parse_args()
    validate_args(args)

    use_half = args.half and str(args.device).lower() != "cpu"

    return AppConfig(
        input_path=args.input,
        output_dir=args.output,
        model=args.model,
        confidence=args.conf,
        image_size=args.imgsz,
        device=str(args.device),
        batch_size=args.batch,
        use_half=use_half,
        real_seconds_per_frame=args.real_seconds_per_frame,
        window_seconds=args.window_seconds,
        threshold=args.threshold,
        pre_seconds=args.pre_seconds,
        post_seconds=args.post_seconds,
        review_fps=args.review_fps,
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""

    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf must be between 0 and 1")
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than 0")
    if args.batch <= 0:
        raise ValueError("--batch must be greater than 0")
    if args.real_seconds_per_frame <= 0:
        raise ValueError("--real-seconds-per-frame must be greater than 0")
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be greater than 0")
    if args.threshold < 0:
        raise ValueError("--threshold must be 0 or greater")
    if args.pre_seconds < 0:
        raise ValueError("--pre-seconds must be 0 or greater")
    if args.post_seconds < 0:
        raise ValueError("--post-seconds must be 0 or greater")
    if args.review_fps <= 0:
        raise ValueError("--review-fps must be greater than 0")
