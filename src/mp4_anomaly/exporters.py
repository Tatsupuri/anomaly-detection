"""CSV and chart exporters."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .settings import CLASS_NAMES
from .video import format_time


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    """Write rows to a UTF-8 CSV file."""

    if fieldnames is None:
        if not rows:
            return
        fieldnames = list(rows[0])

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


def plot_moving_sums(
    path: Path,
    stats_rows: list[dict[str, object]],
    anomaly_rows: list[dict[str, object]],
    threshold: int,
) -> None:
    """Save the moving-sum chart."""

    if not stats_rows:
        return

    elapsed = [
        float(row["real_elapsed_seconds"])
        for row in stats_rows
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    for name in CLASS_NAMES:
        ax.plot(
            elapsed,
            [
                int(row[f"{name}_moving_sum"])
                for row in stats_rows
            ],
            label=name,
            linewidth=1,
        )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1,
        label=f"threshold={threshold}",
    )

    for event in anomaly_rows:
        ax.axvspan(
            float(event["threshold_start_real_seconds"]),
            float(event["threshold_end_real_seconds"]),
            alpha=0.12,
        )

    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: format_time(value))
    )
    ax.set_xlabel("Real elapsed time")
    ax.set_ylabel("Moving sum")
    ax.set_title("Object detections: moving sum")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
