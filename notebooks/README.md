# Notebooks

Auxiliary notebooks that produced inputs to the benchmark. They are not part of the
benchmark runtime (which runs on the Pi via `scripts/benchmark.sh`).

## `taco_yolo26_training_colab.ipynb`

The Google Colab notebook used to fine-tune the YOLO26 models on the TACO dataset.
Run on a T4 GPU runtime with Google Drive mounted for checkpoint output.

What it does:

1. Mounts Google Drive for checkpoint output.
2. Downloads [TACO](https://github.com/pedropro/TACO) and converts it to YOLO format,
   collapsing every annotation to a single `litter` class (`nc: 1`).
3. Splits 80/10/10 train/val/test with `random.seed(42)`, writes `data.yaml`, and
   exports the test split as `taco_test.zip` (the fixed 150-image test set used by the
   benchmark, deployed to `data/content/taco_yolo/`).
4. Fine-tunes one YOLO26 model for 50 epochs (`imgsz=640`, `batch=16`, `lr0=0.001`,
   `seed=42`) and validates it on the test split.
