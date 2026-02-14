import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Callable
import matplotlib.pyplot as plt


def shift_to_target(results):
    results = results.copy()  # modify by value

    # Shape: (num_samples, num_classes)
    y_pred = results.loc[:, results.columns.str.contains("logit")].values
    y_true = results.loc[:, results.columns.str.contains("score")].values

    # Compute per-class stats
    mu_pred = y_pred.mean(axis=0)
    std_pred = y_pred.std(axis=0)

    mu_true = y_true.mean(axis=0)
    std_true = y_true.std(axis=0)

    # Rescale and shift per class
    y_rescaled = ((y_pred - mu_pred) / std_pred) * std_true + mu_true
    results.loc[:, results.columns.str.contains("logit")] = y_rescaled

    return results


def compute_metrics(
    metrics,
    classes,
    df,
    grouping_cols: List,
    target_col_name="targets",
    pred_col_name="logit",
):
    groups_buffer = [
        [{"name": [], "group": df}],
    ]

    for grouping_col in grouping_cols:
        news = []

        for old in groups_buffer[-1]:
            new = [
                {"name": old["name"] + [name], "group": group}
                for name, group in old["group"].groupby(grouping_col)
            ]
            news.extend(new)

        groups_buffer.append(news)

    res = []
    # iterate on the last layer of grouping and compute metrics
    for e in groups_buffer[-1]:

        target = torch.tensor(
            e["group"].loc[:, e["group"].columns.str.contains(target_col_name)].values
        )
        pred = torch.tensor(
            e["group"].loc[:, e["group"].columns.str.contains(pred_col_name)].values
        )

        # get rid of dummy dimension in integer targets
        if target.shape[-1] == 1:
            target = target.squeeze()

        for name, metric_func in metrics.items():
            result = metric_func(pred, target)

            if isinstance(result, torch.Tensor):
                result = result.numpy()

            if result.ndim == 1:
                res_dict = {"metric": name}
                res_dict.update(dict(zip(grouping_cols, e["name"])))
                res_dict.update(dict(zip(classes, result)))
                res.append(res_dict)

            else:
                clean_name, average_kind = name.split("_")
                # res_dict = {"metric":}
                # result = torch.tile(result, (5,))
                metric_idx = [
                    i
                    for i, elem in enumerate(res)
                    if (elem["metric"] == clean_name)
                    & (elem["seed"] == e["name"][1])
                    & (elem["fold"] == e["name"][0])
                ]
                assert len(metric_idx) == 1, "Metric index should be unique"

                res[metric_idx[0]]["average"] = result

    return pd.DataFrame(res)


def find_best_seed_per_fold(df, classes, metric, stat="max"):
    # Filter for the given metric
    df = df[df["metric"] == metric].copy()

    # Compute unweighted class average
    df["avg"] = df[classes].mean(axis=1)

    # Collect best seed per fold
    best = []
    for fold, group in df.groupby("fold"):
        if stat == "max":
            best_row = group.loc[group["avg"].idxmax()]
        elif stat == "min":
            best_row = group.loc[group["avg"].idxmin()]
        else:
            raise ValueError("Unsupported stat. Use 'max' or 'min'.")
        best.append((fold, best_row["seed"]))

    return best
