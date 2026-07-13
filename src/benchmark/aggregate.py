from __future__ import annotations

import csv
import argparse
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src.benchmark.config import RESULTS_DIR

GRID_DIR = Path(RESULTS_DIR) / "grid"

OUTPUT_NAME_MARKERS = ("summary",)

SUMMARY_FIELDS = [
    "strategy",
    "network_latency_ms",
    "n_images",
    "n_latency_samples",
    "n_errors",
    "accuracy_source_latency_ms",
    "mean_latency_ms",
    "median_latency_ms",
    "p95_latency_ms",
    "std_latency_ms",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "offload_rate",
    "zero_detection_offload_share",
]


@dataclass
class GroupStats:
    latencies: list[float] = field(default_factory=list)
    offloaded_count: int = 0
    zero_detection_offload_count: int = 0
    error_count: int = 0


def _is_ok(row: dict[str, str]) -> bool:
    return row.get("status", "ok") != "error"


def input_csv_files(input_dir: Path) -> list[Path]:
    return [
        f for f in sorted(input_dir.glob("*.csv"))
        if not any(marker in f.name for marker in OUTPUT_NAME_MARKERS)
    ]


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    rank = max(1, -(-len(ordered) * 95 // 100))
    return ordered[rank - 1]


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _image_key(row: dict[str, str]) -> str:
    return row.get("image_filename") or row.get("image") or ""


def _accuracy_rows_by_strategy(csv_files: list[Path]) -> dict[str, tuple[int, list[dict[str, str]]]]:
    candidates: dict[str, dict[int, dict[str, dict[str, str]]]] = {}

    for csv_path in csv_files:
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if not _is_ok(row):
                    continue
                strategy = row["strategy"]
                latency = int(row["network_latency_ms"])
                image_key = _image_key(row)
                if not image_key:
                    continue

                latency_candidates = candidates.setdefault(strategy, {})
                image_rows = latency_candidates.setdefault(latency, {})
                image_rows.setdefault(image_key, row)

    accuracy_rows: dict[str, tuple[int, list[dict[str, str]]]] = {}
    for strategy, latency_candidates in candidates.items():
        preferred_latency = 0 if 0 in latency_candidates else min(latency_candidates)
        chosen_rows = latency_candidates[preferred_latency]
        accuracy_rows[strategy] = (preferred_latency, list(chosen_rows.values()))

    return accuracy_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate benchmark CSV results into a summary CSV")
    parser.add_argument("--input-dir", default=str(GRID_DIR), help="Directory containing benchmark CSV files")
    parser.add_argument("--output", default=None, help="Summary CSV output path (defaults to <input-dir>/summary.csv)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    summary_path = Path(args.output) if args.output else input_dir / "summary.csv"

    csv_files = input_csv_files(input_dir)

    if not csv_files:
        print(f"ERROR: no CSV files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    groups: dict[tuple[str, int], GroupStats] = {}
    accuracy_rows = _accuracy_rows_by_strategy(csv_files)

    for csv_path in csv_files:
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                key = (
                    row["strategy"],
                    int(row["network_latency_ms"]),
                )
                g = groups.setdefault(key, GroupStats())
                if not _is_ok(row):
                    g.error_count += 1
                    continue
                g.latencies.append(float(row["latency_ms"]))
                if row["offloaded"].strip().lower() == "true":
                    g.offloaded_count += 1
                    if float(row["local_confidence"]) == 0.0:
                        g.zero_detection_offload_count += 1

    summary_rows: list[dict] = []
    for (strategy, lat), g in sorted(groups.items()):
        n = len(g.latencies)
        accuracy_source_latency_ms, accuracy_rows_for_strategy = accuracy_rows.get(strategy, (lat, []))
        tp = sum(int(row["tp"]) for row in accuracy_rows_for_strategy)
        if n == 0:
            mean_lat = median_lat = p95_lat = std_lat = 0.0
        else:
            mean_lat = round(statistics.mean(g.latencies), 1)
            median_lat = round(statistics.median(g.latencies), 1)
            p95_lat = round(_p95(g.latencies), 1)
            std_lat = round(statistics.stdev(g.latencies), 1) if n > 1 else 0.0
        fp = sum(int(row["fp"]) for row in accuracy_rows_for_strategy)
        fn = sum(int(row["fn"]) for row in accuracy_rows_for_strategy)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        is_hi = strategy in ("hi-edge", "hi-cloud")
        offload_rate = g.offloaded_count / n if is_hi and n > 0 else None
        zero_detection_share = (
            g.zero_detection_offload_count / g.offloaded_count if is_hi and g.offloaded_count > 0 else None
        )

        summary_rows.append({
            "strategy": strategy,
            "network_latency_ms": lat,
            "n_images": len(accuracy_rows_for_strategy),
            "n_latency_samples": n,
            "n_errors": g.error_count,
            "accuracy_source_latency_ms": accuracy_source_latency_ms,
            "mean_latency_ms": mean_lat,
            "median_latency_ms": median_lat,
            "p95_latency_ms": p95_lat,
            "std_latency_ms": std_lat,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(_f1(precision, recall), 4),
            "offload_rate": round(offload_rate, 4) if offload_rate is not None else "n/a",
            "zero_detection_offload_share": (
                round(zero_detection_share, 4) if zero_detection_share is not None else "n/a"
            ),
        })

    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    col_w = [16, 8, 8, 11, 13, 12, 12, 9, 8, 6, 12]
    headers = [
        "strategy",
        "lat_ms",
        "n_img",
        "mean_lat_ms",
        "median_lat_ms",
        "p95_lat_ms",
        "std_lat_ms",
        "precision",
        "recall",
        "f1",
        "offload_rate",
    ]
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))
    print(header_line)
    print("-" * len(header_line))
    for row in summary_rows:
        print("  ".join(str(row[f]).ljust(w) for f, w in zip(
            [
                "strategy",
                "network_latency_ms",
                "n_images",
                "mean_latency_ms",
                "median_latency_ms",
                "p95_latency_ms",
                "std_latency_ms",
                "precision",
                "recall",
                "f1",
                "offload_rate",
            ],
            col_w,
        )))

    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
