from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import tensorflow as tf

from ecg_ml.config import PATHS


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    epochs: int = 10
    validation_split: float = 0.2
    learning_rate: float = 1e-3


def build_model(input_shape: Tuple[int, ...], num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 1000)(inputs)
    x = tf.keras.layers.Conv1D(32, 7, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(64, 5, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool1D(2)(x)
    x = tf.keras.layers.Conv1D(128, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    config: TrainConfig,
) -> tf.keras.Model:
    model = build_model((features.shape[1], 1), int(labels.max() + 1))
    model.fit(
        features[..., np.newaxis],
        labels,
        batch_size=config.batch_size,
        epochs=config.epochs,
        validation_split=config.validation_split,
    )
    return model


def save_model(model: tf.keras.Model, label_map: dict, name: str) -> Path:
    PATHS.models.mkdir(parents=True, exist_ok=True)
    model_path = PATHS.models / f"{name}.keras"
    model.save(model_path)
    joblib.dump(label_map, PATHS.models / f"{name}_labels.joblib")
    return model_path


def load_model(name: str) -> Tuple[tf.keras.Model, dict]:
    model = tf.keras.models.load_model(PATHS.models / f"{name}.keras")
    label_map = joblib.load(PATHS.models / f"{name}_labels.joblib")
    return model, label_map
