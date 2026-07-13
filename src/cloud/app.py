
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import uvicorn

from src.inference.server import create_app

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "yolo26m.pt"

app = create_app(title="Cloud Inference Service", model_path=str(MODEL_PATH))


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
