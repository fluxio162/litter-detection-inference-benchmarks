from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src.benchmark.config import MODEL_PATH, RESULTS_DIR, TEST_IMAGES_DIR, TEST_LABELS_DIR
from src.device.app import IMAGE_EXTENSIONS, REMOTE_URLS, Strategy, infer, warm_up
from src.inference.model import YOLOModel
from src.utils.metrics import compute_image_ap, load_ground_truth

STRATEGIES: tuple[Strategy, ...] = ("device", "edge", "cloud", "hi-edge", "hi-cloud")
NEEDS_LOCAL_MODEL: frozenset[Strategy] = frozenset({"device", "hi-edge", "hi-cloud"})

CSV_FIELDS = [
    "image",
    "run_index",
    "strategy",
    "network_latency_ms",
    "status",
    "error",
    "offloaded",
    "latency_ms",
    "device_inference_ms",
    "remote_request_ms",
    "server_inference_ms",
    "network_ms",
    "bytes_sent",
    "bytes_received",
    "local_confidence",
    "tp",
    "fp",
    "fn",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed ML inference benchmark runner")
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--output", default=None, help="Output CSV path (overrides default naming)")
    parser.add_argument("--run-index", type=int, default=1, metavar="N", help="Repetition index for this run")
    parser.add_argument("--network-latency", type=int, default=0, metavar="MS",
                        help="Injected network latency in ms (for logging only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy: Strategy = args.strategy
    run_index: int = args.run_index
    network_latency_ms: int = args.network_latency

    images_dir = Path(TEST_IMAGES_DIR)
    if not images_dir.exists():
        print(f"ERROR: TEST_IMAGES_DIR not found: {images_dir}", file=sys.stderr)
        sys.exit(1)

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        print(f"ERROR: No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        csv_path = Path(args.output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        csv_path = Path(RESULTS_DIR) / f"{strategy}_results.csv"

    model: YOLOModel | None = None
    if strategy in NEEDS_LOCAL_MODEL:
        print(f"Loading local model from {MODEL_PATH} …")
        model = YOLOModel(model_path=str(MODEL_PATH))

    session = requests.Session()
    if strategy in REMOTE_URLS:
        print(f"Warming up connection to {REMOTE_URLS[strategy]} …")
        warm_up(session, strategy, image_paths[0].read_bytes())

    print(f"Strategy  : {strategy}")
    print(f"Run index : {run_index}")
    print(f"Network   : latency={network_latency_ms}ms")
    print(f"Images    : {len(image_paths)} found in {images_dir}")
    print(f"Output    : {csv_path}")
    print()

    error_count = 0

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        fh.flush()

        for image_path in image_paths:
            base_row = {
                "image": image_path.name,
                "run_index": run_index,
                "strategy": strategy,
                "network_latency_ms": network_latency_ms,
            }

            t_start = time.perf_counter()
            try:
                result = infer(model, image_path.read_bytes(), strategy, session)
            except Exception as exc:
                error_count += 1
                failed_latency_ms = (time.perf_counter() - t_start) * 1000
                writer.writerow({
                    **base_row,
                    "status": "error",
                    "error": str(exc),
                    "offloaded": False,
                    "latency_ms": round(failed_latency_ms, 3),
                    "device_inference_ms": 0.0,
                    "remote_request_ms": 0.0,
                    "server_inference_ms": 0.0,
                    "network_ms": 0.0,
                    "bytes_sent": 0,
                    "bytes_received": 0,
                    "local_confidence": 0.0,
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                })
                fh.flush()
                print(f"  ERROR {image_path.name}: {exc}", file=sys.stderr)
                continue

            label_path = Path(TEST_LABELS_DIR) / (image_path.stem + ".txt")
            gt_boxes = load_ground_truth(label_path)
            ap = compute_image_ap(result.detections, gt_boxes)

            writer.writerow({
                **base_row,
                "status": "ok",
                "error": "",
                "offloaded": result.offloaded,
                "latency_ms": result.latency_ms,
                "device_inference_ms": result.device_inference_ms,
                "remote_request_ms": result.remote_request_ms,
                "server_inference_ms": result.server_inference_ms,
                "network_ms": result.network_ms,
                "bytes_sent": result.bytes_sent,
                "bytes_received": result.bytes_received,
                "local_confidence": result.local_confidence,
                "tp": ap.tp,
                "fp": ap.fp,
                "fn": ap.fn,
            })
            fh.flush()

            if strategy in ("hi-edge", "hi-cloud"):
                offload_tag = " [offloaded]" if result.offloaded else " [local]"
                print(
                    f"  {image_path.name:<40} "
                    f"{result.latency_ms:7.0f} ms  "
                    f"det={result.num_detections:3d}  "
                    f"local_conf={result.local_confidence:.3f} → "
                    f"final_conf={result.max_confidence:.3f}  "
                    f"P={ap.precision:.2f} R={ap.recall:.2f}"
                    f"{offload_tag}"
                )
            else:
                print(
                    f"  {image_path.name:<40} "
                    f"{result.latency_ms:7.1f} ms  "
                    f"det={result.num_detections:3d}  "
                    f"conf={result.max_confidence:.3f}  "
                    f"P={ap.precision:.2f} R={ap.recall:.2f}"
                )

    print(f"\nDone. Results written to {csv_path}")
    if error_count:
        print(f"WARNING: {error_count} image(s) failed and were logged with status=error", file=sys.stderr)


if __name__ == "__main__":
    main()
