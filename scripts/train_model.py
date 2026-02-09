from __future__ import annotations

import argparse

from ecg_ml.data import build_label_map, combine_datasets, encode_labels, load_dataset
from ecg_ml.model import TrainConfig, save_model, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG classification model.")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--model-name", default="ecg_cnn")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = [load_dataset(path) for path in args.datasets]
    features, labels = combine_datasets(datasets)
    label_map = build_label_map(labels)
    encoded_labels = encode_labels(labels, label_map)

    model = train_model(
        features,
        encoded_labels,
        TrainConfig(batch_size=args.batch_size, epochs=args.epochs),
    )
    save_model(model, label_map.to_dict(), args.model_name)


if __name__ == "__main__":
    main()
