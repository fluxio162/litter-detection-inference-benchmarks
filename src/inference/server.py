
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference.model import YOLOModel


def create_app(title: str, model_path: str | None = None) -> FastAPI:
    model: YOLOModel | None = None

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        nonlocal model
        model = YOLOModel(model_path=model_path)
        yield

    app = FastAPI(
        title=title,
        description="Inference API for edge/cloud tiers. Use /health for liveness and /infer for image inference.",
        lifespan=lifespan,
    )

    @app.get(
        "/health",
        summary="Health Check",
        description="Returns server liveness status. Use this to verify the API is reachable.",
        response_description="Simple status payload.",
    )
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/infer",
        summary="Run Inference",
        description="Accepts one uploaded image file and returns model inference output.",
        response_description="Inference result payload.",
    )
    async def infer(file: UploadFile = File(...)) -> dict[str, object]:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        image_bytes = await file.read()
        result = model.predict(image=image_bytes)
        result["bytes_transferred"] = len(image_bytes)
        return result

    return app
