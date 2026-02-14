import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
import argparse
from pathlib import Path
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv_dataset",
    type=str,
    required=True,
)
parser.add_argument(
    "--out_dir",
    type=str,
    required=True,
)
parser.add_argument(
    "--n_splits",
    type=int,
    default=5,
)
parser.add_argument(
    "--no_train",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--no_test",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--random_state",
    type=int,
    default=42,
)
parser.add_argument(
    "--label_col",
    type=str,
    default="subtype",
)
parser.add_argument(
    "--id_col",
    type=str,
    default="slide_id",
)
parser.add_argument(
    "--classes",
    type=str,
    nargs="+",
)
parser.add_argument(
    "--unstable_diff_threshold",
    type=float,
    default=None,
)
parser.add_argument(
    "--classification",
    type=str,
)
parser.add_argument(
    "--samples_to_exclude",
    type=str,
    nargs="+",
    default=[],
    help="List of sample IDs to exclude",
)
parser.add_argument(
    "--stratify_scores",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--aggregation_dict",
    type=json.loads,
    default=None,
)

args = parser.parse_args()
args.csv_dataset = Path(args.csv_dataset)
args.out_dir = Path(args.out_dir)


def compute_stratified_labels(
    metadata,
    classes,
    label_col,
    n_bins=2,
):
    # Calcola la std tra le 5 classi per ciascun sample
    metadata["score_std"] = metadata[classes].std(axis=1)

    # Binna la score_std in quantili
    metadata["score_bin"] = pd.qcut(
        metadata["score_std"], q=n_bins, duplicates="drop"
    ).astype(str)

    # Crea etichetta combinata: classe + bin
    strat_labels = metadata[label_col].astype(str) + "_" + metadata["score_bin"]

    return strat_labels


def split(args):
    metadata = pd.read_csv(args.csv_dataset)

    print(f"Dropping {metadata[args.label_col].isna().sum()} NA rows...")
    metadata = metadata[~metadata[args.label_col].isna()]

    samples_to_exclude = len(args.samples_to_exclude)
    if samples_to_exclude > 0:
        excluded_data = metadata[
            metadata[args.id_col].astype(str).isin(args.samples_to_exclude)
        ]

        print(f"Trying to exclude {samples_to_exclude} samples...")
        print(f"Found and excluded {excluded_data.shape[0]} samples...")
        print(excluded_data[args.label_col].value_counts())

        metadata = metadata[
            ~metadata[args.id_col].astype(str).isin(args.samples_to_exclude)
        ]

    # drop rows where top2 diff scores is lower than threshold
    if args.unstable_diff_threshold:
        print("Filtering out unstable samples...")
        scores = metadata.loc[:, metadata.columns.isin(args.classes)]
        sorted_scores = np.partition(scores.values, -2, axis=1)[:, -2:]
        scores_diffs = np.abs(sorted_scores[:, 1] - sorted_scores[:, 0])
        stable_samples = scores_diffs > args.unstable_diff_threshold
        print(
            f"Retaining {stable_samples.sum()} stable samples (out of {scores.shape[0]})..."
        )
        metadata = metadata.loc[stable_samples]

    # retain scores and target cols, id cols and surv cols
    metadata = metadata[
        args.classes
        + [args.label_col]
        + metadata.columns[metadata.columns.str.contains("surv|id")].tolist()
    ]

    if args.aggregation_dict is not None:
        print(args.aggregation_dict)
        print("Aggregating labels...")
        metadata[args.label_col] = metadata[args.label_col].map(args.aggregation_dict)
        print(metadata)

    if args.stratify_scores:
        metadata["label_strat"] = compute_stratified_labels(
            metadata,
            args.classes,
            args.label_col,
        )
        args.label_col = "label_strat"

    X = metadata
    y = metadata[args.label_col]

    if args.n_splits == 1:
        if args.no_test:
            split_out_dir = (
                args.out_dir.parent / (args.out_dir.stem + "_full_train") / "fold_1"
            )
            split_out_dir.mkdir(parents=True, exist_ok=True)
            print("\nUsing all samples for train/val (no splitting) (FINAL TRAINING)")
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.1,
                stratify=y,
                random_state=42,
            )
            X_train.to_csv(split_out_dir / "train.csv", index=False)
            X_test.to_csv(split_out_dir / "val.csv", index=False)
            X_test.iloc[0:0].to_csv(split_out_dir / "test.csv", index=False)
        elif args.no_train:
            split_out_dir = (
                args.out_dir.parent / (args.out_dir.stem + "_full_test") / "fold_1"
            )
            split_out_dir.mkdir(parents=True, exist_ok=True)
            X.iloc[0:0].to_csv(split_out_dir / "train.csv", index=False)
            X.iloc[0:0].to_csv(split_out_dir / "val.csv", index=False)
            X.to_csv(split_out_dir / "test.csv", index=False)

        else:
            split_out_dir = args.out_dir / "fold_0"
            split_out_dir.mkdir(parents=True, exist_ok=True)

            X.to_csv(split_out_dir / "train.csv", index=False)
            X.to_csv(split_out_dir / "val.csv", index=False)
            X.to_csv(split_out_dir / "test.csv", index=False)

            print(f"Train/Val/Test size: {len(X)}, {X[args.label_col].value_counts()}")

    else:
        skf = StratifiedKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.random_state,
        )

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\nOuter fold {fold + 1}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            split_out_dir = args.out_dir / f"fold_{fold + 1}"
            split_out_dir.mkdir(parents=True, exist_ok=True)

            # Save ONLY outer splits
            X_train.to_csv(split_out_dir / "train.csv", index=False)
            X_test.to_csv(split_out_dir / "test.csv", index=False)

            print(
                f"Train size: {len(X_train)}, {X_train[args.label_col].value_counts()}\n"
                f"Test size:  {len(X_test)}, {X_test[args.label_col].value_counts()}"
            )


if __name__ == "__main__":
    split(args)
