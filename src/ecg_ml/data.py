from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import wfdb

from ecg_ml.config import PATHS


@dataclass(frozen=True)
class SegmentConfig:
    sample_rate: int = 360
    segment_seconds: float = 2.0
    lead: int = 0

    @property
    def segment_samples(self) -> int:
        return int(self.sample_rate * self.segment_seconds)


def _ensure_dirs() -> None:
    PATHS.raw.mkdir(parents=True, exist_ok=True)
    PATHS.processed.mkdir(parents=True, exist_ok=True)


def download_mit_bih(records: Iterable[str]) -> List[Path]:
    _ensure_dirs()
    downloaded: List[Path] = []
    for record in records:
        wfdb.dl_database("mitdb", str(PATHS.raw), records=[record])
        downloaded.append(PATHS.raw / record)
    return downloaded


def download_chapman(records: Iterable[str]) -> List[Path]:
    _ensure_dirs()
    downloaded: List[Path] = []
    for record in records:
        wfdb.dl_database("chapman_shaoxing", str(PATHS.raw), records=[record])
        downloaded.append(PATHS.raw / record)
    return downloaded


def load_mit_bih_segments(records: Iterable[str], config: SegmentConfig) -> Tuple[np.ndarray, np.ndarray]:
    segments: List[np.ndarray] = []
    labels: List[str] = []
    half = config.segment_samples // 2
    for record in records:
        record_path = PATHS.raw / record
        signal, fields = wfdb.rdsamp(str(record_path))
        ann = wfdb.rdann(str(record_path), "atr")
        for sample, symbol in zip(ann.sample, ann.symbol):
            start = max(sample - half, 0)
            end = start + config.segment_samples
            if end > signal.shape[0]:
                continue
            segment = signal[start:end, config.lead]
            segments.append(segment)
            labels.append(symbol)
    return np.stack(segments), np.array(labels)


def _parse_chapman_label(header: wfdb.Record) -> str:
    comments = header.comments or []
    for comment in comments:
        if comment.startswith("Dx"):
            return comment.split(":", 1)[-1].strip()
    return "Unknown"


def load_chapman_segments(records: Iterable[str], config: SegmentConfig) -> Tuple[np.ndarray, np.ndarray]:
    segments: List[np.ndarray] = []
    labels: List[str] = []
    for record in records:
        record_path = PATHS.raw / record
        signal, fields = wfdb.rdsamp(str(record_path))
        header = wfdb.rdheader(str(record_path))
        label = _parse_chapman_label(header)
        samples = signal.shape[0]
        stride = config.segment_samples
        for start in range(0, samples - config.segment_samples + 1, stride):
            segment = signal[start : start + config.segment_samples, config.lead]
            segments.append(segment)
            labels.append(label)
    return np.stack(segments), np.array(labels)


def save_dataset(features: np.ndarray, labels: np.ndarray, name: str) -> Path:
    _ensure_dirs()
    path = PATHS.processed / f"{name}.npz"
    np.savez_compressed(path, X=features, y=labels)
    return path


def load_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"]


def combine_datasets(datasets: Iterable[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = zip(*datasets)
    return np.concatenate(xs), np.concatenate(ys)


def build_label_map(labels: np.ndarray) -> pd.Series:
    unique = pd.Series(labels).unique()
    return pd.Series({label: idx for idx, label in enumerate(sorted(unique))})


def encode_labels(labels: np.ndarray, label_map: pd.Series) -> np.ndarray:
    return np.array([label_map[label] for label in labels], dtype=np.int64)
