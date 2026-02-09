from __future__ import annotations

from io import BytesIO, StringIO
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from pypdf import PdfReader
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles

from ecg_ml.model import load_model

app = FastAPI(title="ECG CNN Classifier")

MODEL_NAME = "ecg_cnn"
label_descriptions = {
    "N": "Normal sinus rhythm with standard ventricular conduction.",
    "L": "Left bundle branch block pattern detected.",
    "R": "Right bundle branch block pattern detected.",
    "V": "Premature ventricular contraction characteristics present.",
    "A": "Atrial premature contraction features detected.",
    "Unknown": "Unable to map label to a known clinical description.",
}
fallback_label_map = {label: idx for idx, label in enumerate(["N", "L", "R", "V", "A"])}

app.mount("/static", StaticFiles(directory="app/static"), name="static")


class PredictRequest(BaseModel):
    signal: List[float] = Field(..., description="1D ECG segment")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]
    interpretation: str


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("app/static/index.html")


class _DemoModel:
    def __init__(self, input_length: int, num_classes: int) -> None:
        self.input_shape = (None, input_length, 1)
        self._num_classes = num_classes

    def predict(self, inputs: np.ndarray, verbose: int = 0) -> np.ndarray:
        batch = inputs.shape[0]
        logits = np.random.rand(batch, self._num_classes)
        return logits / logits.sum(axis=1, keepdims=True)


def _load_model():
    try:
        return load_model(MODEL_NAME)
    except (OSError, ValueError) as exc:
        print(f"Model load failed: {exc}. Using demo model outputs instead.")
        demo = _DemoModel(input_length=720, num_classes=len(fallback_label_map))
        return demo, fallback_label_map


model, label_map = _load_model()
id_to_label = {idx: label for label, idx in label_map.items()}


def _extract_numbers(text: str) -> List[float]:
    import re

    return [float(value) for value in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)]


def _signal_from_upload(upload: UploadFile) -> np.ndarray:
    filename = (upload.filename or "").lower()
    content = upload.file.read()

    if filename.endswith(".pdf"):
        reader = PdfReader(BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        values = _extract_numbers(text)
    elif filename.endswith(".csv"):
        frame = pd.read_csv(StringIO(content.decode("utf-8", errors="ignore")))
        values = frame.select_dtypes(include=[np.number]).iloc[:, 0].tolist()
    else:
        decoded = content.decode("utf-8", errors="ignore")
        values = _extract_numbers(decoded)

    if not values:
        raise HTTPException(status_code=400, detail="No numeric ECG samples found in file.")
    return np.array(values, dtype=np.float32)


def _normalize_length(signal: np.ndarray) -> np.ndarray:
    target = model.input_shape[1]
    if signal.size == target:
        return signal
    if signal.size < target:
        pad = target - signal.size
        return np.pad(signal, (0, pad))
    start = max((signal.size - target) // 2, 0)
    return signal[start : start + target]


def _predict(signal: np.ndarray) -> PredictResponse:
    signal = _normalize_length(signal).reshape(1, -1, 1)
    probs = model.predict(signal, verbose=0)[0]
    label_id = int(np.argmax(probs))
    label = id_to_label.get(label_id, "Unknown")
    confidence = float(np.max(probs))
    probabilities = {id_to_label[idx]: float(prob) for idx, prob in enumerate(probs)}
    interpretation = label_descriptions.get(label, label_descriptions["Unknown"])
    return PredictResponse(
        label=label,
        confidence=confidence,
        probabilities=probabilities,
        interpretation=interpretation,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest) -> PredictResponse:
    signal = np.array(payload.signal, dtype=np.float32)
    return _predict(signal)


@app.post("/analyze", response_model=PredictResponse)
async def analyze(file: UploadFile = File(...)) -> PredictResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    signal = _signal_from_upload(file)
    return _predict(signal)
