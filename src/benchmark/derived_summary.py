from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

CELLS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("cloud", (0, 100, 300)),
    ("device", (0,)),
    ("edge", (0, 100, 300)),
    ("hi-cloud", (0, 100, 300)),
    ("hi-edge", (0, 100, 300)),
)

OUTPUT_NAME = "derived_summary.csv"

FIELDNAMES = [
    "strategy",
    "network_latency_ms",
    "n",
    "p5_ms",
    "q1_ms",
    "q3_ms",
    "rep_mean_1",
    "rep_mean_2",
    "rep_mean_3",
    "rep_mean_4",
    "rep_mean_5",
    "median_device_inference_ms",
]


@dataclass
class RunRecords:
    latencies: list[float] = field(default_factory=list)
    device_inference: list[float] = field(default_factory=list)


def interpolated_percentile(sorted_values: list[float], percent: float) -> float:
    position = (len(sorted_values) - 1) * percent / 100
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def load_cell(input_dir: Path, strategy: str, delay: int) -> dict[int, RunRecords]:
    runs: dict[int, RunRecords] = {}
    for path in sorted(input_dir.glob(f"{strategy}_lat{delay}ms_run*.csv")):
        run_index = int(path.stem.rsplit("run", 1)[1])
        records = runs.setdefault(run_index, RunRecords())
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status", "ok") not in ("", "ok"):
                    continue
                records.latencies.append(float(row["latency_ms"]))
                device_ms = row.get("device_inference_ms") or ""
                if device_ms and float(device_ms) > 0:
                    records.device_inference.append(float(device_ms))
    return runs


def cell_row(strategy: str, delay: int, runs: dict[int, RunRecords]) -> dict[str, object]:
    pooled = sorted(latency for records in runs.values() for latency in records.latencies)
    device_values = [value for records in runs.values() for value in records.device_inference]
    row: dict[str, object] = {
        "strategy": strategy,
        "network_latency_ms": delay,
        "n": len(pooled),
        "p5_ms": round(interpolated_percentile(pooled, 5), 1),
        "q1_ms": round(interpolated_percentile(pooled, 25), 1),
        "q3_ms": round(interpolated_percentile(pooled, 75), 1),
        "median_device_inference_ms": round(statistics.median(device_values), 1) if device_values else "",
    }
    for run_index in range(1, 6):
        records = runs.get(run_index)
        row[f"rep_mean_{run_index}"] = round(statistics.fmean(records.latencies), 1) if records else ""
    return row


def write_derived_summary(input_dir: Path) -> Path:
    output_path = input_dir / OUTPUT_NAME
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for strategy, delays in CELLS:
            for delay in delays:
                writer.writerow(cell_row(strategy, delay, load_cell(input_dir, strategy, delay)))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    arguments = parser.parse_args()
    output_path = write_derived_summary(arguments.input_dir)
    print(output_path)


if __name__ == "__main__":
    main()
