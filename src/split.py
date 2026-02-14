from pathlib import Path
import pandas as pd
from typing import Any, Dict, Optional, Tuple
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import hydra
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger, extras



log = RankedLogger(__name__, rank_zero_only=True)


def split(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    metadata = pd.read_csv(cfg.data_split.csv_dataset)

    log.info(f"Dropping {metadata[cfg.data_split.label_col].isna().sum()} NA rows...")
    metadata = metadata[~metadata[cfg.data_split.label_col].isna()]

    # retain scores and target cols, id cols and surv cols
    metadata = metadata[
        cfg.data_split.classes
        + [cfg.data_split.label_col]
        + metadata.columns[metadata.columns.str.contains("surv|id")].tolist()
    ]

    X = metadata
    y = metadata[cfg.data_split.label_col]

    full_out_dir = cfg.data_split.out_dir / "full" / "full"
    full_out_dir.mkdir(exist_ok=True, parents=True)
    metadata.to_csv(full_out_dir / "train.csv")
    
    assert cfg.data_split.outer_folds > 1
    assert cfg.data_split.inner_folds > 1

    outer_skf = StratifiedKFold(
        n_splits=cfg.data_split.outer_folds,
        shuffle=True,
        random_state=cfg.seed,
    )

    for outer_fold, (train_idx_outer, test_idx_outer) in enumerate(
        outer_skf.split(X, y)
    ):
        X_train_outer, X_test_outer = (
            X.iloc[train_idx_outer],
            X.iloc[test_idx_outer],
        )
        y_train_outer, _ = (
            y.iloc[train_idx_outer],
            y.iloc[test_idx_outer],
        )

        # get held out validation set from the training set
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=cfg.data_split.outer_val_frac,  # e.g. 0.1
            random_state=cfg.seed + 5_000 + outer_fold * 100,
        )
        train_fit_outer, train_val_outer = next(
            sss.split(
                X_train_outer,
                y_train_outer,
            )
        )

        X_train_fit_outer = X_train_outer.iloc[train_fit_outer]
        X_train_val_outer = X_train_outer.iloc[train_val_outer]

        outer_fold_out = cfg.data_split.out_dir / str(outer_fold) / "full"
        outer_fold_out.mkdir(exist_ok=True, parents=True)

        X_train_fit_outer.to_csv(outer_fold_out / "full" / "train.csv", index=None)
        X_train_val_outer.to_csv(outer_fold_out / "full" / "val.csv", index=None)
        X_test_outer.to_csv(outer_fold_out / "full" / "test.csv", index=None)

        inner_skf = StratifiedKFold(
            n_splits=cfg.data_split.inner_folds,
            shuffle=True,
            random_state=cfg.seed + outer_fold * 100,
        )

        for inner_fold, (train_idx_inner, test_idx_inner) in enumerate(
            inner_skf.split(X_train_outer, y_train_outer)
        ):
            X_train_inner, X_test_inner = (
                X_train_outer.iloc[train_idx_inner],
                X_train_outer.iloc[test_idx_inner],
            )
            y_train_inner, _ = (
                y_train_outer.iloc[train_idx_inner],
                y_train_outer.iloc[test_idx_inner],
            )

            # get held out validation set from the training set
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=cfg.data_split.outer_val_frac,  # e.g. 0.1
                random_state=cfg.seed + 10_000 + outer_fold * 100 + inner_fold,
            )
            train_fit_inner, train_val_inner = next(
                sss.split(
                    X_train_inner,
                    y_train_inner,
                )
            )

            X_train_fit_inner = X_train_inner.iloc[train_fit_inner]
            X_train_val_inner = X_train_inner.iloc[train_val_inner]

            inner_fold_out = outer_fold_out / str(inner_fold)
            inner_fold_out.mkdir(exist_ok=True)

            X_train_fit_inner.to_csv(inner_fold_out / "train.csv", index=None)
            X_train_val_inner.to_csv(inner_fold_out / "val.csv", index=None)
            X_test_inner.to_csv(inner_fold_out / "test.csv", index=None)

            log.info(
                f"""\n\r
                outer fold: {outer_fold} - inner fold: {inner_fold}
                Training size: {len(X_train_fit_inner)}
                {X_train_fit_inner[cfg.data_split.label_col].value_counts()}
                Validation size: {len(X_train_val_inner)}
                {X_train_val_inner[cfg.data_split.label_col].value_counts()}
                Test size: {len(X_test_inner)}
                {X_test_inner[cfg.data_split.label_col].value_counts()}
            """
            )


@hydra.main(version_base="1.3", config_path="../configs", config_name="split.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)
    print(cfg)
    cfg.data_split.out_dir = Path(cfg.data_split.out_dir)
    cfg.data_split.out_dir.mkdir(exist_ok=True)
    # train the model
    split(cfg)

    # return optimized metric
    return


if __name__ == "__main__":
    main()
