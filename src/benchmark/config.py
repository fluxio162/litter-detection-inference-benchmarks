import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = _REPO_ROOT / "models" / "yolo26n.pt"
TEST_IMAGES_DIR = _REPO_ROOT / "data" / "content" / "taco_yolo" / "test" / "images"
TEST_LABELS_DIR = _REPO_ROOT / "data" / "content" / "taco_yolo" / "test" / "labels"
RESULTS_DIR = _REPO_ROOT / "results"

# Inference endpoints. Set these to your own edge/cloud servers before running the
# benchmark, e.g. `export EDGE_INFER_URL=http://<edge-ip>:8000/infer`.
EDGE_INFER_URL = os.environ.get("EDGE_INFER_URL", "http://localhost:8000/infer")
CLOUD_INFER_URL = os.environ.get("CLOUD_INFER_URL", "http://localhost:8001/infer")

HI_CONFIDENCE_THRESHOLD = 0.8
