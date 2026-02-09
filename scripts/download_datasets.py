from __future__ import annotations

import argparse

from ecg_ml.data import (
    SegmentConfig,
    download_chapman,
    download_mit_bih,
    load_chapman_segments,
    load_mit_bih_segments,
    save_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and preprocess ECG datasets.")
    parser.add_argument("--mit-records", nargs="*", default=["100", "101", "102"])
    parser.add_argument("--chapman-records", nargs="*", default=["A0001", "A0002"])
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=360)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SegmentConfig(sample_rate=args.sample_rate, segment_seconds=args.segment_seconds)

    download_mit_bih(args.mit_records)
    download_chapman(args.chapman_records)

    mit_features, mit_labels = load_mit_bih_segments(args.mit_records, config)
    chap_features, chap_labels = load_chapman_segments(args.chapman_records, config)

    save_dataset(mit_features, mit_labels, "mit_bih")
    save_dataset(chap_features, chap_labels, "chapman")


if __name__ == "__main__":
    main()
