
import time

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOModel:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.model(np.zeros((640, 640, 3), dtype=np.uint8), conf=0.25, verbose=False)

    def predict(
        self, image: bytes, confidence_threshold: float = 0.25
    ) -> dict[str, object]:
        image_array = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

        if image_array is None:
            raise ValueError(
                "Could not decode image — unsupported format or corrupt bytes"
            )

        start = time.perf_counter()
        results = self.model(image_array, conf=confidence_threshold, verbose=False)
        inference_ms = (time.perf_counter() - start) * 1000

        detections: list[dict[str, object]] = []
        for result in results:
            for box in result.boxes:
                xywhn = box.xywhn[0].tolist()
                detections.append(
                    {
                        "x_center": xywhn[0],
                        "y_center": xywhn[1],
                        "width": xywhn[2],
                        "height": xywhn[3],
                        "confidence": float(box.conf[0]),
                        "class_name": result.names[int(box.cls[0])],
                    }
                )

        return {
            "detections": detections,
            "inference_ms": inference_ms,
        }
