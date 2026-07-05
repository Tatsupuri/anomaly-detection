from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import matplotlib

# 画面を開かずにグラフを保存する
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from ultralytics import YOLO


# COCOクラス
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


@dataclass
class FramePacket:
    """
    確認動画へ書き込む1フレーム分の情報。
    """

    frame_number: int
    frame: Any

    # class_name, confidence, (x1, y1, x2, y2)
    detections: list[
        tuple[
            str,
            float,
            tuple[int, int, int, int],
        ]
    ]

    moving_sums: dict[str, int]


@dataclass
class EventState:
    """
    1つの異常区間の情報。
    """

    event_id: int
    clip_path: Path
    clip_start_frame: int
    threshold_start_frame: int
    last_anomaly_frame: int
    clip_end_frame: int

    trigger_classes: set[str] = field(
        default_factory=set
    )

    peak_class: str = ""
    peak_moving_sum: int = 0


def format_time(seconds: float) -> str:
    """
    秒数を HH:MM:SS に変換する。
    """

    total = max(
        0,
        int(round(seconds)),
    )

    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{secs:02d}"
    )


def annotate(
    packet: FramePacket,
    real_seconds_per_frame: float,
    source_fps: float,
) -> Any:
    """
    元画像に以下を描画する。

    ・検出枠
    ・クラス名
    ・信頼度
    ・現実時間換算
    ・元MP4上の再生時間
    ・クラス別移動和
    """

    image = packet.frame.copy()

    for (
        class_name,
        confidence,
        (x1, y1, x2, y2),
    ) in packet.detections:
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            image,
            f"{class_name} {confidence:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    real_time = format_time(
        packet.frame_number
        * real_seconds_per_frame
    )

    source_time = format_time(
        packet.frame_number
        / source_fps
    )

    sums = " ".join(
        f"{name}={packet.moving_sums[name]}"
        for name in CLASS_NAMES
    )

    lines = [
        (
            f"real elapsed: {real_time}  "
            f"source video: {source_time}"
        ),
        f"moving sum: {sums}",
    ]

    for index, text in enumerate(lines):
        cv2.putText(
            image,
            text,
            (12, 28 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return image


def open_writer(
    path: Path,
    width: int,
    height: int,
    fps: float,
) -> cv2.VideoWriter:
    """
    MP4書き込みを開始する。
    """

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"動画を書き込めません: {path}"
        )

    return writer


def write_packet(
    writer: cv2.VideoWriter,
    packet: FramePacket,
    real_seconds_per_frame: float,
    source_fps: float,
    review_fps: float,
) -> None:
    """
    1フレームを確認動画へ書き込む。

    タイムラプスの1フレームを、
    現実の撮影間隔に相当する時間だけ表示するため、
    同じ画像を複数回書き込む。
    """

    image = annotate(
        packet,
        real_seconds_per_frame,
        source_fps,
    )

    repeat = max(
        1,
        round(
            review_fps
            * real_seconds_per_frame
        ),
    )

    for _ in range(repeat):
        writer.write(image)


def extract_valid_detections(
    result: Any,
) -> tuple[
    dict[str, int],
    list[
        tuple[
            str,
            float,
            tuple[int, int, int, int],
        ]
    ],
    int,
]:
    """
    1フレームの検出結果を安全に変換する。

    以下は不正な検出として無視する。

    ・NaN
    ・正負の無限大
    ・対象外のクラスID
    ・幅または高さが0の枠

    戻り値:
      counts
      detections
      invalid_count
    """

    counts = {
        name: 0
        for name in CLASS_NAMES
    }

    detections: list[
        tuple[
            str,
            float,
            tuple[int, int, int, int],
        ]
    ] = []

    invalid_count = 0

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):
        return (
            counts,
            detections,
            invalid_count,
        )

    frame_height, frame_width = (
        result.orig_img.shape[:2]
    )

    for box in result.boxes:
        class_value = float(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        coordinates = (
            box.xyxy[0]
            .detach()
            .cpu()
            .tolist()
        )

        values = [
            class_value,
            confidence,
            *coordinates,
        ]

        # NaN、Infinity、不正な座標数を除外
        if (
            len(coordinates) != 4
            or not all(
                math.isfinite(value)
                for value in values
            )
        ):
            invalid_count += 1
            continue

        class_id = int(class_value)

        # 対象外または不明なクラスを除外
        if class_id not in CLASS_ID_TO_NAME:
            invalid_count += 1
            continue

        x1, y1, x2, y2 = map(
            round,
            coordinates,
        )

        # 座標を画像範囲内に制限
        x1 = max(
            0,
            min(
                x1,
                frame_width - 1,
            ),
        )

        y1 = max(
            0,
            min(
                y1,
                frame_height - 1,
            ),
        )

        x2 = max(
            0,
            min(
                x2,
                frame_width - 1,
            ),
        )

        y2 = max(
            0,
            min(
                y2,
                frame_height - 1,
            ),
        )

        # 幅または高さがない枠を除外
        if x2 <= x1 or y2 <= y1:
            invalid_count += 1
            continue

        class_name = (
            CLASS_ID_TO_NAME[class_id]
        )

        counts[class_name] += 1

        detections.append(
            (
                class_name,
                confidence,
                (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
            )
        )

    return (
        counts,
        detections,
        invalid_count,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "YOLOのカテゴリー別検出数の移動和を計算し、"
            "閾値超過区間を枠付き動画として保存します。"
        )
    )

    parser.add_argument(
        "input",
        type=Path,
        help="入力MP4",
    )

    parser.add_argument(
        "output",
        type=Path,
        help="出力ディレクトリ",
    )

    parser.add_argument(
        "--model",
        default="yolo11m.pt",
        help="YOLOモデル",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="検出信頼度の下限",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="推論時の画像サイズ",
    )

    parser.add_argument(
        "--device",
        default="0",
        help="GPUは0、CPUはcpu",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="推論バッチサイズ",
    )

    parser.add_argument(
        "--half",
        action="store_true",
        help=(
            "FP16で推論する。"
            "NaNが多い場合は外してください。"
        ),
    )

    parser.add_argument(
        "--real-seconds-per-frame",
        type=float,
        default=2.0,
        help=(
            "入力動画の1フレームが"
            "現実の何秒に相当するか"
        ),
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=60.0,
        help="移動和の時間幅",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
        help=(
            "移動和がこの値を超えると異常"
        ),
    )

    parser.add_argument(
        "--pre-seconds",
        type=float,
        default=5.0,
        help="異常開始前に保存する秒数",
    )

    parser.add_argument(
        "--post-seconds",
        type=float,
        default=5.0,
        help="異常終了後に保存する秒数",
    )

    parser.add_argument(
        "--review-fps",
        type=float,
        default=10.0,
        help="確認動画のFPS",
    )

    return parser


def validate_args(
    args: argparse.Namespace,
) -> None:
    """
    コマンドライン引数を検証する。
    """

    if not args.input.exists():
        raise FileNotFoundError(
            f"入力動画がありません: {args.input}"
        )

    if args.real_seconds_per_frame <= 0:
        raise ValueError(
            "--real-seconds-per-frame は"
            "0より大きくしてください"
        )

    if args.window_seconds <= 0:
        raise ValueError(
            "--window-seconds は"
            "0より大きくしてください"
        )

    if args.pre_seconds < 0:
        raise ValueError(
            "--pre-seconds は"
            "0以上にしてください"
        )

    if args.post_seconds < 0:
        raise ValueError(
            "--post-seconds は"
            "0以上にしてください"
        )

    if args.review_fps <= 0:
        raise ValueError(
            "--review-fps は"
            "0より大きくしてください"
        )

    if args.threshold < 0:
        raise ValueError(
            "--threshold は"
            "0以上にしてください"
        )

    if args.batch <= 0:
        raise ValueError(
            "--batch は"
            "1以上にしてください"
        )


def read_video_info(
    input_path: Path,
) -> tuple[
    float,
    int,
    int,
    int,
]:
    """
    元MP4のFPS、総フレーム数、幅、高さを取得する。
    """

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"動画を開けません: {input_path}"
        )

    source_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    cap.release()

    if source_fps <= 0:
        raise RuntimeError(
            "入力動画のFPSを取得できません"
        )

    if width <= 0 or height <= 0:
        raise RuntimeError(
            "入力動画の解像度を取得できません"
        )

    return (
        source_fps,
        total_frames,
        width,
        height,
    )


def finish_event(
    writer: cv2.VideoWriter,
    event: EventState,
    source_fps: float,
    args: argparse.Namespace,
    anomaly_rows: list[
        dict[str, object]
    ],
) -> None:
    """
    異常動画を閉じてログへ追加する。
    """

    writer.release()

    anomaly_rows.append(
        finalize_event(
            event,
            source_fps,
            args,
        )
    )


def main() -> None:
    args = make_parser().parse_args()

    validate_args(args)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    clips_dir = (
        args.output / "clips"
    )

    clips_dir.mkdir(
        exist_ok=True
    )

    (
        source_fps,
        total_frames,
        width,
        height,
    ) = read_video_info(
        args.input
    )

    # 60秒 ÷ 2秒 = 30フレーム
    window_frames = max(
        1,
        math.ceil(
            args.window_seconds
            / args.real_seconds_per_frame
        ),
    )

    # 5秒 ÷ 2秒 = 2.5
    # 切り上げて3フレーム
    pre_frames = max(
        0,
        math.ceil(
            args.pre_seconds
            / args.real_seconds_per_frame
        ),
    )

    post_frames = max(
        0,
        math.ceil(
            args.post_seconds
            / args.real_seconds_per_frame
        ),
    )

    # CPUではFP16を使わない
    use_half = (
        args.half
        and str(args.device).lower()
        != "cpu"
    )

    if args.half and not use_half:
        print(
            "[警告] CPUでは"
            "--halfを無効化します。"
        )

    print(f"入力: {args.input}")
    print(
        f"出力: {args.output.resolve()}"
    )
    print(f"モデル: {args.model}")

    print(
        f"移動和: "
        f"{args.window_seconds:g}秒 "
        f"({window_frames}フレーム)"
    )

    print(
        f"異常条件: "
        f"いずれかのカテゴリーの移動和 "
        f"> {args.threshold}"
    )

    print(
        f"切り出し: "
        f"前{pre_frames}フレーム、"
        f"後{post_frames}フレーム"
    )

    print(f"FP16: {use_half}")

    model = YOLO(args.model)

    results = model.predict(
        source=str(args.input),

        # 逐次処理
        stream=True,

        classes=CLASS_IDS,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        half=use_half,

        # 全フレーム処理
        vid_stride=1,

        verbose=False,
    )

    # 直近の移動窓に含まれる
    # フレームごとの検出数
    rolling_counts: deque[
        dict[str, int]
    ] = deque()

    # 現在のクラス別移動和
    rolling_sums = {
        name: 0
        for name in CLASS_NAMES
    }

    # 異常開始前のフレームを保持
    pre_buffer: deque[
        FramePacket
    ] = deque(
        maxlen=pre_frames + 1
    )

    # 全フレームの統計
    stats_rows: list[
        dict[str, object]
    ] = []

    # 異常区間のログ
    anomaly_rows: list[
        dict[str, object]
    ] = []

    writer: (
        cv2.VideoWriter | None
    ) = None

    event: (
        EventState | None
    ) = None

    event_id = 0
    post_remaining = 0
    last_written_frame = -1
    invalid_detections = 0

    for frame_number, result in tqdm(
        enumerate(results),
        total=total_frames or None,
        unit="frame",
    ):
        (
            counts,
            detections,
            invalid_count,
        ) = extract_valid_detections(
            result
        )

        invalid_detections += (
            invalid_count
        )

        # 現在フレームの検出数を
        # 移動窓へ追加
        rolling_counts.append(
            counts
        )

        for name in CLASS_NAMES:
            rolling_sums[name] += (
                counts[name]
            )

        # 窓を超えた古いフレームを除外
        if (
            len(rolling_counts)
            > window_frames
        ):
            removed = (
                rolling_counts.popleft()
            )

            for name in CLASS_NAMES:
                rolling_sums[name] -= (
                    removed[name]
                )

        # 60秒分のフレームが
        # そろっているか
        window_ready = (
            len(rolling_counts)
            == window_frames
        )

        packet = FramePacket(
            frame_number=frame_number,
            frame=result.orig_img.copy(),
            detections=detections,
            moving_sums=dict(
                rolling_sums
            ),
        )

        pre_buffer.append(packet)

        real_seconds = (
            frame_number
            * args.real_seconds_per_frame
        )

        source_seconds = (
            frame_number
            / source_fps
        )

        stats_rows.append(
            {
                "frame_number":
                    frame_number,

                "real_elapsed_seconds":
                    real_seconds,

                "real_elapsed_time":
                    format_time(
                        real_seconds
                    ),

                "source_video_seconds":
                    source_seconds,

                "source_video_time":
                    format_time(
                        source_seconds
                    ),

                "window_ready":
                    window_ready,

                **counts,

                **{
                    f"{name}_moving_sum":
                        rolling_sums[name]
                    for name
                    in CLASS_NAMES
                },
            }
        )

        triggered: list[str] = []

        # 60秒分そろってから異常判定
        if window_ready:
            triggered = [
                name
                for name in CLASS_NAMES
                if (
                    rolling_sums[name]
                    > args.threshold
                )
            ]

        is_anomaly = bool(triggered)

        # 新しい異常区間を開始
        if (
            writer is None
            and is_anomaly
        ):
            event_id += 1

            clip_path = (
                clips_dir
                / (
                    f"anomaly_"
                    f"{event_id:04d}.mp4"
                )
            )

            writer = open_writer(
                clip_path,
                width,
                height,
                args.review_fps,
            )

            clip_start_frame = (
                pre_buffer[0]
                .frame_number
            )

            # 異常開始前のフレームを
            # 動画へ書き込む
            for buffered in pre_buffer:
                write_packet(
                    writer,
                    buffered,
                    args.real_seconds_per_frame,
                    source_fps,
                    args.review_fps,
                )

            # 現在フレームは
            # pre_bufferに含まれている
            last_written_frame = (
                frame_number
            )

            peak_class = max(
                CLASS_NAMES,
                key=lambda name:
                    rolling_sums[name],
            )

            event = EventState(
                event_id=event_id,
                clip_path=clip_path,
                clip_start_frame=(
                    clip_start_frame
                ),
                threshold_start_frame=(
                    frame_number
                ),
                last_anomaly_frame=(
                    frame_number
                ),
                clip_end_frame=(
                    frame_number
                ),
                trigger_classes=set(
                    triggered
                ),
                peak_class=peak_class,
                peak_moving_sum=(
                    rolling_sums[
                        peak_class
                    ]
                ),
            )

            post_remaining = (
                post_frames
            )

            continue

        # 異常動画を記録中
        if (
            writer is not None
            and event is not None
        ):
            if is_anomaly:
                if (
                    frame_number
                    > last_written_frame
                ):
                    write_packet(
                        writer,
                        packet,
                        args.real_seconds_per_frame,
                        source_fps,
                        args.review_fps,
                    )

                    last_written_frame = (
                        frame_number
                    )

                event.clip_end_frame = (
                    frame_number
                )

                event.last_anomaly_frame = (
                    frame_number
                )

                event.trigger_classes.update(
                    triggered
                )

                current_peak_class = max(
                    CLASS_NAMES,
                    key=lambda name:
                        rolling_sums[name],
                )

                current_peak_value = (
                    rolling_sums[
                        current_peak_class
                    ]
                )

                if (
                    current_peak_value
                    > event.peak_moving_sum
                ):
                    event.peak_class = (
                        current_peak_class
                    )

                    event.peak_moving_sum = (
                        current_peak_value
                    )

                # 異常が継続したので
                # 後方フレーム数をリセット
                post_remaining = (
                    post_frames
                )

            elif post_remaining > 0:
                # 異常終了後のフレームを保存
                if (
                    frame_number
                    > last_written_frame
                ):
                    write_packet(
                        writer,
                        packet,
                        args.real_seconds_per_frame,
                        source_fps,
                        args.review_fps,
                    )

                    last_written_frame = (
                        frame_number
                    )

                event.clip_end_frame = (
                    frame_number
                )

                post_remaining -= 1

                if post_remaining == 0:
                    finish_event(
                        writer,
                        event,
                        source_fps,
                        args,
                        anomaly_rows,
                    )

                    writer = None
                    event = None

            else:
                # post-secondsが0の場合
                finish_event(
                    writer,
                    event,
                    source_fps,
                    args,
                    anomaly_rows,
                )

                writer = None
                event = None

    # 動画末尾まで異常が続いた場合
    if (
        writer is not None
        and event is not None
    ):
        finish_event(
            writer,
            event,
            source_fps,
            args,
            anomaly_rows,
        )

    moving_sum_csv = (
        args.output
        / "moving_sum.csv"
    )

    anomalies_csv = (
        args.output
        / "anomalies.csv"
    )

    moving_sum_png = (
        args.output
        / "moving_sum.png"
    )

    write_stats_csv(
        moving_sum_csv,
        stats_rows,
    )

    write_anomalies_csv(
        anomalies_csv,
        anomaly_rows,
    )

    write_graph(
        moving_sum_png,
        stats_rows,
        anomaly_rows,
        args.threshold,
    )

    print()
    print(
        f"異常区間数: "
        f"{len(anomaly_rows)}"
    )

    print(
        f"無視した不正検出数: "
        f"{invalid_detections}"
    )

    print(
        f"確認動画: "
        f"{clips_dir.resolve()}"
    )

    print(
        f"グラフ: "
        f"{moving_sum_png.resolve()}"
    )

    print(
        f"異常ログ: "
        f"{anomalies_csv.resolve()}"
    )

    print(
        f"全フレーム統計: "
        f"{moving_sum_csv.resolve()}"
    )

    if (
        invalid_detections > 0
        and use_half
    ):
        print(
            "[注意] FP16でNaNなどの"
            "不正な検出値が発生しています。"
        )

        print(
            "不正検出が多い場合は、"
            "--halfを外して再実行してください。"
        )


def finalize_event(
    event: EventState,
    source_fps: float,
    args: argparse.Namespace,
) -> dict[str, object]:
    """
    異常区間をCSV用の辞書に変換する。
    """

    def real_seconds(
        frame: int,
    ) -> float:
        return (
            frame
            * args.real_seconds_per_frame
        )

    def source_seconds(
        frame: int,
    ) -> float:
        return frame / source_fps

    return {
        "event_id":
            event.event_id,

        "trigger_classes":
            ",".join(
                sorted(
                    event.trigger_classes
                )
            ),

        "peak_class":
            event.peak_class,

        "peak_moving_sum":
            event.peak_moving_sum,

        "clip_start_frame":
            event.clip_start_frame,

        "clip_end_frame":
            event.clip_end_frame,

        "threshold_start_frame":
            event.threshold_start_frame,

        "last_anomaly_frame":
            event.last_anomaly_frame,

        "clip_start_real_seconds":
            real_seconds(
                event.clip_start_frame
            ),

        "clip_start_real_time":
            format_time(
                real_seconds(
                    event.clip_start_frame
                )
            ),

        "clip_end_real_seconds":
            real_seconds(
                event.clip_end_frame
            ),

        "clip_end_real_time":
            format_time(
                real_seconds(
                    event.clip_end_frame
                )
            ),

        "threshold_start_real_time":
            format_time(
                real_seconds(
                    event.threshold_start_frame
                )
            ),

        "last_anomaly_real_time":
            format_time(
                real_seconds(
                    event.last_anomaly_frame
                )
            ),

        "clip_start_source_seconds":
            source_seconds(
                event.clip_start_frame
            ),

        "clip_start_source_time":
            format_time(
                source_seconds(
                    event.clip_start_frame
                )
            ),

        "clip_end_source_seconds":
            source_seconds(
                event.clip_end_frame
            ),

        "clip_end_source_time":
            format_time(
                source_seconds(
                    event.clip_end_frame
                )
            ),

        "clip_path":
            str(event.clip_path),
    }


def write_stats_csv(
    path: Path,
    rows: list[
        dict[str, object]
    ],
) -> None:
    """
    全フレームの検出数と移動和を保存する。
    """

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def write_anomalies_csv(
    path: Path,
    rows: list[
        dict[str, object]
    ],
) -> None:
    """
    異常区間一覧を保存する。
    """

    fieldnames = [
        "event_id",
        "trigger_classes",
        "peak_class",
        "peak_moving_sum",
        "clip_start_frame",
        "clip_end_frame",
        "threshold_start_frame",
        "last_anomaly_frame",
        "clip_start_real_seconds",
        "clip_start_real_time",
        "clip_end_real_seconds",
        "clip_end_real_time",
        "threshold_start_real_time",
        "last_anomaly_real_time",
        "clip_start_source_seconds",
        "clip_start_source_time",
        "clip_end_source_seconds",
        "clip_end_source_time",
        "clip_path",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_graph(
    path: Path,
    stats_rows: list[
        dict[str, object]
    ],
    anomaly_rows: list[
        dict[str, object]
    ],
    threshold: int,
) -> None:
    """
    クラス別移動和グラフを保存する。
    """

    if not stats_rows:
        return

    x = [
        float(
            row[
                "real_elapsed_seconds"
            ]
        )
        for row in stats_rows
    ]

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    for name in CLASS_NAMES:
        ax.plot(
            x,
            [
                int(
                    row[
                        f"{name}_moving_sum"
                    ]
                )
                for row
                in stats_rows
            ],
            label=name,
            linewidth=1,
        )

    # 閾値
    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1,
        label=(
            f"threshold={threshold}"
        ),
    )

    # 異常として保存した区間
    for event in anomaly_rows:
        ax.axvspan(
            float(
                event[
                    "clip_start_real_seconds"
                ]
            ),
            float(
                event[
                    "clip_end_real_seconds"
                ]
            ),
            alpha=0.12,
        )

    ax.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _:
                format_time(value)
        )
    )

    ax.set_xlabel(
        "Real elapsed time"
    )

    ax.set_ylabel(
        "Moving sum"
    )

    ax.set_title(
        "Object detections: moving sum"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=150,
    )

    plt.close(fig)


if __name__ == "__main__":
    try:
        main()

    except Exception:
        import traceback

        traceback.print_exc()

        input(
            "Enterキーを押して"
            "終了してください..."
        )