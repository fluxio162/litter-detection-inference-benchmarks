from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class APResult:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float


def load_ground_truth(label_path: Path) -> list[dict]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, x_center, y_center, width, height = parts
        boxes.append({
            "x_center": float(x_center),
            "y_center": float(y_center),
            "width": float(width),
            "height": float(height),
        })
    return boxes


def compute_iou(box_a: dict, box_b: dict) -> float:
    def to_xyxy(box: dict) -> tuple[float, float, float, float]:
        x1 = box["x_center"] - box["width"] / 2
        y1 = box["y_center"] - box["height"] / 2
        x2 = box["x_center"] + box["width"] / 2
        y2 = box["y_center"] + box["height"] / 2
        return x1, y1, x2, y2

    ax1, ay1, ax2, ay2 = to_xyxy(box_a)
    bx1, by1, bx2, by2 = to_xyxy(box_b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area == 0.0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter_area / (area_a + area_b - inter_area)


def compute_image_ap(
    pred_boxes: list[dict],
    gt_boxes: list[dict],
    iou_threshold: float = 0.5,
) -> APResult:
    sorted_preds = sorted(pred_boxes, key=lambda b: b["confidence"], reverse=True)
    matched_gt: set[int] = set()
    tp = 0

    for pred in sorted_preds:
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(sorted_preds) - tp
    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if sorted_preds else 0.0
    recall = tp / (tp + fn) if gt_boxes else 0.0

    return APResult(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall)
