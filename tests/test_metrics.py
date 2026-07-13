import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from src.utils.metrics import compute_image_ap, compute_iou, load_ground_truth


def box(x_center: float, y_center: float, width: float, height: float, confidence: float | None = None) -> dict:
    result = {"x_center": x_center, "y_center": y_center, "width": width, "height": height}
    if confidence is not None:
        result["confidence"] = confidence
    return result


class TestComputeIou:
    def test_identical_boxes(self) -> None:
        a = box(0.5, 0.5, 0.2, 0.2)
        assert compute_iou(a, a) == pytest.approx(1.0)

    def test_disjoint_boxes(self) -> None:
        assert compute_iou(box(0.2, 0.2, 0.1, 0.1), box(0.8, 0.8, 0.1, 0.1)) == 0.0

    def test_touching_boxes_have_zero_iou(self) -> None:
        assert compute_iou(box(0.2, 0.5, 0.2, 0.2), box(0.4, 0.5, 0.2, 0.2)) == 0.0

    def test_half_overlap(self) -> None:
        a = box(0.4, 0.5, 0.2, 0.2)
        b = box(0.5, 0.5, 0.2, 0.2)
        assert compute_iou(a, b) == pytest.approx(1.0 / 3.0)

    def test_contained_box(self) -> None:
        outer = box(0.5, 0.5, 0.4, 0.4)
        inner = box(0.5, 0.5, 0.2, 0.2)
        assert compute_iou(outer, inner) == pytest.approx(0.25)


class TestComputeImageAp:
    def test_perfect_match(self) -> None:
        gt = [box(0.5, 0.5, 0.2, 0.2)]
        preds = [box(0.5, 0.5, 0.2, 0.2, confidence=0.9)]
        result = compute_image_ap(preds, gt)
        assert (result.tp, result.fp, result.fn) == (1, 0, 0)
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_no_predictions_with_ground_truth(self) -> None:
        result = compute_image_ap([], [box(0.5, 0.5, 0.2, 0.2)])
        assert (result.tp, result.fp, result.fn) == (0, 0, 1)
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_predictions_without_ground_truth(self) -> None:
        result = compute_image_ap([box(0.5, 0.5, 0.2, 0.2, confidence=0.9)], [])
        assert (result.tp, result.fp, result.fn) == (0, 1, 0)
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_empty_image(self) -> None:
        result = compute_image_ap([], [])
        assert (result.tp, result.fp, result.fn) == (0, 0, 0)

    def test_duplicate_predictions_on_one_ground_truth(self) -> None:
        gt = [box(0.5, 0.5, 0.2, 0.2)]
        preds = [
            box(0.5, 0.5, 0.2, 0.2, confidence=0.9),
            box(0.5, 0.5, 0.2, 0.2, confidence=0.8),
        ]
        result = compute_image_ap(preds, gt)
        assert (result.tp, result.fp, result.fn) == (1, 1, 0)

    def test_higher_confidence_prediction_matched_first(self) -> None:
        gt = [box(0.5, 0.5, 0.2, 0.2)]
        preds = [
            box(0.52, 0.5, 0.2, 0.2, confidence=0.6),
            box(0.5, 0.5, 0.2, 0.2, confidence=0.9),
        ]
        result = compute_image_ap(preds, gt)
        assert (result.tp, result.fp, result.fn) == (1, 1, 0)

    def test_iou_below_threshold_is_false_positive(self) -> None:
        gt = [box(0.3, 0.5, 0.2, 0.2)]
        preds = [box(0.44, 0.5, 0.2, 0.2, confidence=0.9)]
        result = compute_image_ap(preds, gt, iou_threshold=0.5)
        assert (result.tp, result.fp, result.fn) == (0, 1, 1)

    def test_iou_above_threshold_is_true_positive(self) -> None:
        gt = [box(0.5, 0.5, 0.2, 0.2)]
        preds = [box(0.55, 0.5, 0.2, 0.2, confidence=0.9)]
        result = compute_image_ap(preds, gt, iou_threshold=0.5)
        assert (result.tp, result.fp, result.fn) == (1, 0, 0)

    def test_two_objects_two_matches(self) -> None:
        gt = [box(0.25, 0.25, 0.2, 0.2), box(0.75, 0.75, 0.2, 0.2)]
        preds = [
            box(0.25, 0.25, 0.2, 0.2, confidence=0.8),
            box(0.75, 0.75, 0.2, 0.2, confidence=0.7),
        ]
        result = compute_image_ap(preds, gt)
        assert (result.tp, result.fp, result.fn) == (2, 0, 0)


class TestLoadGroundTruth:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert load_ground_truth(tmp_path / "missing.txt") == []

    def test_parses_yolo_lines(self, tmp_path: Path) -> None:
        label = tmp_path / "image.txt"
        label.write_text("0 0.5 0.4 0.3 0.2\n0 0.1 0.2 0.05 0.05\n")
        boxes = load_ground_truth(label)
        assert len(boxes) == 2
        assert boxes[0] == {"x_center": 0.5, "y_center": 0.4, "width": 0.3, "height": 0.2}

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        label = tmp_path / "image.txt"
        label.write_text("0 0.5 0.4 0.3\n\n0 0.5 0.4 0.3 0.2\n")
        assert len(load_ground_truth(label)) == 1
