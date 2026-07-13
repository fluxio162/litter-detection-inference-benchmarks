from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from src.benchmark.config import (
    CLOUD_INFER_URL,
    EDGE_INFER_URL,
    HI_CONFIDENCE_THRESHOLD,
)
from src.inference.model import YOLOModel

HTTP_TIMEOUT = 30
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
Strategy = Literal["device", "edge", "cloud", "hi-edge", "hi-cloud"]

REMOTE_URLS: dict[Strategy, str] = {
    "edge": EDGE_INFER_URL,
    "cloud": CLOUD_INFER_URL,
    "hi-edge": EDGE_INFER_URL,
    "hi-cloud": CLOUD_INFER_URL,
}


@dataclass
class RemoteResponse:
    detections: list[dict]
    server_inference_ms: float
    request_ms: float
    bytes_sent: int
    bytes_received: int


@dataclass
class InferenceResult:
    offloaded: bool
    latency_ms: float
    device_inference_ms: float
    remote_request_ms: float
    server_inference_ms: float
    network_ms: float
    bytes_sent: int
    bytes_received: int
    detections: list[dict]
    num_detections: int
    local_confidence: float
    max_confidence: float


def _max_confidence(detections: list[dict]) -> float:
    return max((d["confidence"] for d in detections), default=0.0)


def _validate_detection_format(detections: list[dict], url: str) -> None:
    if detections and "x_center" not in detections[0]:
        raise RuntimeError(
            f"Server at {url} returned detections in old format (missing 'x_center'). "
            "Restart the server with the updated src/inference/model.py."
        )


def _remote_infer(session: requests.Session, url: str, image_bytes: bytes) -> RemoteResponse:
    start = time.perf_counter()
    response = session.post(
        url,
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
        timeout=HTTP_TIMEOUT,
    )
    request_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    payload = response.json()
    detections: list[dict] = payload.get("detections", [])
    _validate_detection_format(detections, url)
    return RemoteResponse(
        detections=detections,
        server_inference_ms=float(payload.get("inference_ms", 0.0)),
        request_ms=request_ms,
        bytes_sent=len(image_bytes),
        bytes_received=len(response.content),
    )


def warm_up(session: requests.Session, strategy: Strategy, image_bytes: bytes) -> None:
    url = REMOTE_URLS.get(strategy)
    if url is None:
        return
    try:
        session.post(
            url,
            files={"file": ("warmup.jpg", image_bytes, "image/jpeg")},
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException:
        pass


def infer(
    model: YOLOModel | None,
    image_bytes: bytes,
    strategy: Strategy,
    session: requests.Session | None = None,
) -> InferenceResult:
    device_inference_ms = 0.0
    offloaded = False
    remote: RemoteResponse | None = None

    t0 = time.perf_counter()

    if strategy == "device":
        local_result = model.predict(image_bytes)
        detections: list[dict] = local_result["detections"]
        device_inference_ms = float(local_result["inference_ms"])
        local_confidence = _max_confidence(detections)

    elif strategy in ("edge", "cloud"):
        remote = _remote_infer(session, REMOTE_URLS[strategy], image_bytes)
        detections = remote.detections
        local_confidence = _max_confidence(detections)

    elif strategy in ("hi-edge", "hi-cloud"):
        local_result = model.predict(image_bytes)
        local_detections: list[dict] = local_result["detections"]
        device_inference_ms = float(local_result["inference_ms"])
        local_confidence = _max_confidence(local_detections)

        if local_confidence < HI_CONFIDENCE_THRESHOLD:
            remote = _remote_infer(session, REMOTE_URLS[strategy], image_bytes)
            detections = remote.detections
            offloaded = True
        else:
            detections = local_detections

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    latency_ms = (time.perf_counter() - t0) * 1000

    if remote is not None:
        remote_request_ms = remote.request_ms
        server_inference_ms = remote.server_inference_ms
        network_ms = max(0.0, remote.request_ms - remote.server_inference_ms)
        bytes_sent = remote.bytes_sent
        bytes_received = remote.bytes_received
    else:
        remote_request_ms = server_inference_ms = network_ms = 0.0
        bytes_sent = bytes_received = 0

    return InferenceResult(
        offloaded=offloaded,
        latency_ms=latency_ms,
        device_inference_ms=device_inference_ms,
        remote_request_ms=remote_request_ms,
        server_inference_ms=server_inference_ms,
        network_ms=network_ms,
        bytes_sent=bytes_sent,
        bytes_received=bytes_received,
        detections=detections,
        num_detections=len(detections),
        local_confidence=local_confidence,
        max_confidence=_max_confidence(detections),
    )
