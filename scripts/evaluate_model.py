from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from ecg_ml.data import build_label_map, combine_datasets, encode_labels, load_dataset
from ecg_ml.model import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained ECG model.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--model-name", default="ecg_cnn")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [load_dataset(path) for path in args.datasets]
    features, labels = combine_datasets(datasets)

    model, label_map = load_model(args.model_name)
    expected_map = build_label_map(labels)
    if set(label_map.keys()) != set(expected_map.keys()):
        missing = set(expected_map.keys()) - set(label_map.keys())
        extra = set(label_map.keys()) - set(expected_map.keys())
        raise ValueError(
            "Label mismatch between dataset and model. "
            f"Missing in model: {sorted(missing)}. Extra in model: {sorted(extra)}."
        )

    encoded_labels = encode_labels(labels, expected_map)
    probs = model.predict(features[..., np.newaxis], verbose=0)
    preds = np.argmax(probs, axis=1)

    accuracy = accuracy_score(encoded_labels, preds)
    report = classification_report(encoded_labels, preds, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}")
    print(report)


if __name__ == "__main__":
    main()
