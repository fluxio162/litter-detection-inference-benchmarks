# Models

The three fine-tuned YOLO26 weights used by the benchmark. Each was trained for this
project's use case — **single-class litter detection** — on the
[TACO](https://github.com/pedropro/TACO) dataset converted to YOLO format (every annotation
collapsed to a single `litter` class), using the Colab notebook in
[`../notebooks/`](../notebooks/).

| File         | Model            | Tier                        |
| ------------ | ---------------- | --------------------------- |
| `yolo26n.pt` | YOLO26n (nano)   | Device (Raspberry Pi)       |
| `yolo26s.pt` | YOLO26s (small)  | Edge (Lenovo Mini PC)       |
| `yolo26m.pt` | YOLO26m (medium) | Cloud (Hugging Face Space)  |

All three share the same training setup (50 epochs, `imgsz=640`, `seed=42`) and differ only
in model size — which is exactly what the benchmark compares across the device–edge–cloud
tiers.

The weight files are **not committed** to the repo. Reproduce them with the notebook and drop
`yolo26n.pt`, `yolo26s.pt`, and `yolo26m.pt` into this folder (see the main
[README](../README.md#reproducing)).
