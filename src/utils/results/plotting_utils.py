from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import h5py
import re
from shapely import Polygon
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from matplotlib.lines import Line2D
from shapely import Point
from sklearn.calibration import LabelEncoder
from sklearn.metrics import roc_curve, auc
from PIL import Image
import cv2

# from lifelines import KaplanMeierFitter
# from lifelines.statistics import multivariate_logrank_test
# from lifelines.plotting import add_at_risk_counts
from itertools import repeat
from scipy.stats import rankdata
import json

import openslide
from PIL import Image
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from typing import List, Tuple, Callable
from anndata import AnnData
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import seaborn as sns

# import shap


def render_mpl_table(
    data,
    col_width=3.0,
    row_height=0.625,
    font_size=14,
    header_color="#40466e",
    row_colors=["#f1f1f2", "w"],
    edge_color="w",
    bbox=[0, 0, 1, 1],
    header_columns=0,
    ax=None,
    decimals=None,
    **kwargs,
):
    # Copy data to avoid modifying original
    data = data.copy()

    # Format numeric values with fixed number of decimals
    if decimals is not None:
        fmt = f"{{:.{decimals}f}}"
        data = data.applymap(
            lambda x: fmt.format(x) if isinstance(x, (int, float, np.floating)) else x
        )

    # Insert index as first column
    data.insert(0, "", data.index.astype(str))

    if ax is None:
        size = (col_width * data.shape[1], row_height * (data.shape[0] + 1))
        fig, ax = plt.subplots(figsize=size)
        ax.axis("off")

    # Use tab10 colormap for column headers (excluding index header)
    tab10 = plt.get_cmap("tab10")
    header_colors = [
        header_color,
    ] * (
        1 + header_columns
    ) + [mcolors.to_hex(tab10(i % 10)) for i in range(data.shape[1] - 1)]

    mpl_table = ax.table(
        cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs
    )

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for (row, col), cell in mpl_table.get_celld().items():
        cell.set_edgecolor(edge_color)
        if row == 0:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor(header_colors[col])
        else:
            cell.set_facecolor(row_colors[row % len(row_colors)])

    plt.close(fig)  # <---- this line prevents implicit display
    return ax


def plot_contingency_heatmap(
    contingency_table,
    title,
    fmt=".0f",
    ordered_classes=None,
    xlabel="Second Prediction",
    ylabel="First Prediction",
    figsize=(10, 8),
    font_size=8,
    outfile=None,
    show=True,
    cbar_title="Count",
):
    """
    Plot a heatmap of a contingency table with Chi² test annotation.

    Parameters:
    - contingency_table: pd.DataFrame (output of pd.crosstab)
    - ordered_classes: list of str, used for ordered axis tick labels
    - xlabel, ylabel, title: str
    - outfile: str or Path, if provided the figure is saved there
    - show: bool, whether to display the plot
    """

    # Plot heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        contingency_table,
        annot=True,
        fmt=fmt,
        square=True,
        cmap=plt.get_cmap("viridis"),
        annot_kws={"size": font_size, "weight": "bold"},
        cbar_kws={"shrink": 0.75, "label": cbar_title, "pad": 0.02},
        linewidths=0.5,
        linecolor="gray",
    )

    ax.set_xlabel(xlabel, fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_title(title, fontsize=font_size, pad=20)

    if ordered_classes is not None:
        ax.set_xticks(np.arange(len(ordered_classes)) + 0.5)
        ax.set_xticklabels(
            ordered_classes, fontsize=16, rotation=0, ha="right", rotation_mode="anchor"
        )
        ax.set_yticks(np.arange(len(ordered_classes)) + 0.5)
        ax.set_yticklabels(ordered_classes, fontsize=font_size, rotation=0)
    else:
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=font_size, rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=font_size, rotation=0)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)

    # Colorbar font
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)
    cbar.ax.yaxis.label.set_size(font_size)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def plot_clarity_mann_whitney(
    preds,
    targets,
    scores_diffs,
    out_path,
):
    correct = preds == targets
    clarity = scores_diffs

    clarity_correct = clarity[correct]
    clarity_wrong = clarity[~correct]

    stat, p = mannwhitneyu(clarity_wrong, clarity_correct, alternative="less")
    print(f"Mann-Whitney U test: stat={stat:.3f}, p-value={p:.3e}")

    correct_text = np.array(["incorrect", "correct"])[correct.astype(int)]
    df = pd.DataFrame({"clarity": clarity, "prediction": correct_text})

    palette = {"correct": "green", "incorrect": "red"}

    plt.figure(figsize=(8, 6))
    ax = sns.violinplot(
        data=df, x="prediction", y="clarity", inner="box", palette=palette
    )

    # Titles and labels
    ax.set_title(
        "RNA clarity distribution by prediction correctness", fontsize=20, pad=20
    )
    ax.set_xlabel("Prediction", fontsize=18, labelpad=15)
    ax.set_ylabel("Clarity", fontsize=18, labelpad=15)

    # Axis ticks
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

    # Annotation
    formatted_p = f"(Mann-Whitney U one-tail)\n" + (
        f"p = {p:.4g}" if p >= 1e-4 else f"p < 1e-4"
    )
    ax.text(
        0.5,
        max(df["clarity"]) * 1.05,
        formatted_p,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )

    # Border styling
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def plot_scatter_per_class(
    logits,
    scores,
    classes,
    alpha=0.5,
    xlabel="Logit",
    ylabel="Score",
    title_prefix="Scatterplot",
    out_dir=None,
):

    n_samples, n_classes = logits.shape
    assert n_classes == len(classes)
    colors = sns.color_palette("tab10", n_classes)

    for i, class_name in enumerate(classes):
        fig, ax = plt.subplots(figsize=(6, 6))

        class_logits = logits[:, i]
        class_scores = scores[:, i]

        sns.regplot(
            x=class_logits,
            y=class_scores,
            scatter=True,
            color=colors[i],
            scatter_kws={"alpha": alpha, "edgecolor": None, "linewidths": 0},
            line_kws={"linewidth": 2},
            ax=ax,
        )

        # Square limits
        all_vals = np.concatenate([class_logits, class_scores])
        lims = [-1, 1]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")

        # y = x line
        ax.plot(lims, lims, "--", color="gray", linewidth=1.5, zorder=0)

        ax.set_title(f"{title_prefix} - {class_name}", fontsize=16)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis="both", labelsize=12)

        # Legend
        line_handle = Line2D(
            [0], [0], linestyle="--", color="gray", lw=1.5, label="y = x"
        )
        reg_handle = Line2D(
            [0], [0], linestyle="-", color=colors[i], lw=2, label="Linear fit"
        )
        ax.legend(
            handles=[line_handle, reg_handle],
            loc="upper left",
            fontsize=11,
            frameon=True,
        )

        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1.2)

        plt.tight_layout()

        if out_dir:
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_dir / f"scatter_{class_name}.pdf", dpi=300)
            plt.close()
        else:
            plt.show()


def plot_prediction_errors_per_class(
    logits,
    scores,
    targets,
    classes,
    out_path,
):
    # Inputs
    top_classes = targets  # shape (n_samples,)
    colors = plt.get_cmap("tab10").colors
    symbols = ["★"] * len(top_classes)  # you can also use ['▲', '•', etc.]

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(14, 14), sharex=True)

    # Custom legend entry for the symbol
    star_handle = Line2D(
        [0],
        [0],
        marker="*",
        color=colors[7],
        label="Target Class",
        markersize=10,
        linestyle="None",
    )

    for i, class_name in enumerate(classes):
        class_logits = logits[:, i]
        class_scores = scores[:, i]
        diffs = (class_logits - class_scores) / class_logits
        x = np.arange(len(diffs))

        ax = axes[i]
        ax.bar(x, diffs, width=1.0, color=colors[i], alpha=0.8)
        ax.axhline(0, color="black", linewidth=1)

        # Get dynamic x and y limits
        x_max = x[-1]
        y_max, y_min = ax.get_ylim()

        # Text just beyond the rightmost bar
        x_text = x_max - 20  # adjust offset if needed

        # Add labels to the right
        ax.text(
            x_text,
            y_max * 0.75,
            "Underestimates",
            fontsize=12,
            color="black",
            ha="left",
            va="top",
            fontstyle="italic",
        )

        ax.text(
            x_text,
            y_min * 0.75,
            "  Overestimates",
            fontsize=12,
            color="black",
            ha="left",
            va="bottom",
            fontstyle="italic",
        )

        # Annotate the true class with a symbol
        for xi in x[top_classes == i]:
            bar_val = diffs[xi]
            if bar_val >= 0:
                ax.text(
                    xi,
                    bar_val + 0.01,
                    symbols[xi],
                    color=colors[7],
                    fontsize=10,
                    ha="center",
                    va="bottom",
                )
            else:
                ax.text(
                    xi,
                    bar_val - 0.01,
                    symbols[xi],
                    color=colors[7],
                    fontsize=10,
                    ha="center",
                    va="top",
                )

        # Axes styling
        # ax.set_ylabel(f"{class_name}\nΔ(prediction - target)\n", fontsize=14, labelpad=10)
        ax.set_ylabel(
            f"{class_name}\n"
            + r"$\frac{\mathrm{prediction} - \mathrm{target}}{\mathrm{target}}$",
            fontsize=14,
            labelpad=10,
        )
        ax.tick_params(axis="y", labelsize=12)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        ax.legend(
            handles=[star_handle],
            fontsize=12,
            title_fontsize=16,
            loc="lower left",
            frameon=True,
        )

    # Add to existing legend or create a new one

    # Shared x-axis
    axes[-1].set_xlabel("Sample Index", fontsize=16, labelpad=10)
    axes[-1].tick_params(axis="x", labelsize=12)

    # Title
    fig.suptitle("Prediction Error per Class", fontsize=22, y=0.93)
    plt.savefig(out_path, dpi=300)
    plt.show()


def plot_prediction_error_per_sample(
    logits,
    scores,
    targets,
    preds,
    classes,
    out_path,
):
    errors = (logits - scores) / logits
    prediction_errors = errors[np.arange(errors.shape[0]), preds]

    # Inputs
    colors = plt.get_cmap("tab10").colors
    x = np.arange(len(prediction_errors))  # len = 232
    color_map = np.array(colors)[preds]  # color per sample
    symbols = ["★"] * len(preds)  # or any other symbol
    true_class = targets  # shape (232,)

    fig, ax = plt.subplots(figsize=(14, 7))  # Adjust height as needed

    # Bar plot
    ax.bar(x, prediction_errors, width=1.0, color=color_map, alpha=0.8)
    ax.axhline(0, color="black", linewidth=1)

    # Dynamic annotation bounds
    x_text = x[-1] - 20
    y_max, y_min = ax.get_ylim()
    ax.text(
        x_text,
        y_max * 0.75,
        "Underestimates",
        fontsize=12,
        color="black",
        ha="left",
        va="top",
        fontstyle="italic",
    )
    ax.text(
        x_text,
        y_min * 0.75,
        "  Overestimates",
        fontsize=12,
        color="black",
        ha="left",
        va="bottom",
        fontstyle="italic",
    )

    # Annotate RNA true class with symbol (color fixed to 7th colormap entry)
    for xi in x:
        if true_class[xi] == preds[xi]:
            bar_val = prediction_errors[xi]
            offset = 0.01 if bar_val >= 0 else -0.01
            va = "bottom" if bar_val >= 0 else "top"
            ax.text(
                xi,
                bar_val + offset,
                symbols[xi],
                color=colors[7],
                fontsize=10,
                ha="center",
                va=va,
            )

    # Custom legend for the star
    star_handle = Line2D(
        [0],
        [0],
        marker="*",
        color=colors[7],
        label="RNA subtype",
        markersize=10,
        linestyle="None",
    )
    ax.legend(handles=[star_handle], fontsize=12, loc="lower left", frameon=True)

    # Axes styling
    ax.set_xlabel("Sample Index", fontsize=16, labelpad=10)
    ax.set_ylabel("Δ(prediction - target)", fontsize=14, labelpad=10)
    ax.tick_params(axis="both", labelsize=12)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Build legend handles for predicted classes
    class_labels = classes  # list of class names, length 5
    pred_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color=colors[i],
            label=class_labels[i],
            markersize=10,
            linestyle="None",
        )
        for i in range(len(class_labels))
    ]

    legend1 = ax.legend(
        handles=pred_handles,
        title="Predicted Class",
        title_fontsize=12,
        fontsize=12,
        loc="lower center",
        frameon=True,
        ncol=len(pred_handles),
        bbox_to_anchor=(0.5, 0),  # Adjust as needed
    )
    ax.add_artist(legend1)

    star_handle = Line2D(
        [0],
        [0],
        marker="*",
        color=colors[7],
        markersize=12,
        linestyle="None",
    )

    legend2 = ax.legend(
        handles=[star_handle],
        title="Correct Prediction",
        title_fontsize=12,
        fontsize=12,
        loc="lower left",
        frameon=True,
    )

    # Title
    plt.title("Prediction error of the Highest Scoring Class (AI)", fontsize=20, pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


def plot_single_sample_profile(
    logits,
    scores,
    classes,
    sample_index,
    sample_name,
    colors,
    out_path,
):

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot bars (predicted logits)
    ax.bar(
        x=np.arange(5),
        height=logits[sample_index],
        color=colors[:5],
        alpha=0.8,
        width=0.6,
    )

    # Plot scatter (RNA target scores)
    ax.scatter(
        x=np.arange(5),
        y=scores[sample_index],
        color=colors[:5],
        edgecolor="black",
        s=100,
        marker="o",
    )

    # Axis labels
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(classes, fontsize=13)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_xlabel("Subtype", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)

    # Grid and spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)

    # Custom legend handles
    legend_pred_target = [
        Line2D([0], [0], color="gray", lw=10, label="Predicted (AI)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=10,
            label="Target (RNA)",
        ),
    ]

    legend_classes = [
        Line2D(
            [0],
            [0],
            marker="s",
            color=colors[j],
            label=classes[j],
            markersize=12,
            linestyle="None",
        )
        for j in range(len(classes))
    ]

    # Draw legends
    leg1 = ax.legend(
        handles=legend_pred_target, loc="upper left", frameon=False, fontsize=12
    )
    leg2 = ax.legend(
        handles=legend_classes,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(classes),
        frameon=True,
        fontsize=11,
        title="Classes",
    )

    # Add both legends to the axes
    ax.add_artist(leg1)

    # Title
    ax.set_title(
        f"Predicted and Target Scores for Sample {sample_name}",
        fontsize=16,
        pad=10,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_single_sample_profile_v2(
    logits,
    scores,
    classes,
    sample_index,
    sample_name,
    colors,
    out_path,
    title,
    plot_predicted=True,
    vertical=False,
    zero_line=False,
    font_size=16,
    size=(1000, 500),  # in pixels (width, height)
    separate_legend_path=None,  # path to save separate legend PDF
):
    # Enforce correct figure size in PDF units
    dpi = 300
    figsize = (size[0] / dpi, size[1] / dpi)
    fig, ax = plt.subplots(figsize=figsize)

    # Font sizes (not scaled)
    title_fs = font_size
    label_fs = font_size * 0.875
    tick_fs = font_size * 0.75
    legend_fs = font_size * 0.75
    legend_title_fs = font_size * 0.7
    marker_size = 100

    indices = np.arange(len(classes))

    # === Plot target scores as bars ===
    if not vertical:
        ax.bar(
            x=indices,
            height=scores[sample_index],
            color=colors[: len(classes)],
            alpha=0.8,
            width=0.6,
        )
    else:
        ax.barh(
            y=indices,
            width=scores[sample_index],
            color=colors[: len(classes)],
            alpha=0.8,
            height=0.6,
        )

    # === Plot predicted scores as scatter ===
    if plot_predicted:
        if not vertical:
            ax.scatter(
                x=indices,
                y=logits[sample_index],
                color=colors[: len(classes)],
                edgecolor="black",
                s=marker_size,
                marker="o",
                zorder=3,
            )
        else:
            ax.scatter(
                x=logits[sample_index],
                y=indices,
                color=colors[: len(classes)],
                edgecolor="black",
                s=marker_size,
                marker="o",
                zorder=3,
            )

    # === Axes & grid ===
    if not vertical:
        ax.set_xticks(indices)
        ax.set_xticklabels(classes, fontsize=tick_fs, rotation=45, ha="right")
        ax.set_ylabel("Score", fontsize=label_fs)
        ax.set_xlabel("Subtype", fontsize=label_fs)
        ax.tick_params(axis="y", labelsize=tick_fs)
        if zero_line:
            ax.axhline(0, color="black", linewidth=1.2)
        ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    else:
        ax.set_yticks(indices)
        ax.set_yticklabels(classes, fontsize=tick_fs)
        ax.set_xlabel("Score", fontsize=label_fs)
        ax.set_ylabel("Subtype", fontsize=label_fs)
        ax.tick_params(axis="x", labelsize=tick_fs)
        if zero_line:
            ax.axvline(0, color="black", linewidth=1.2)
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)

    # === Only show in-plot legend if separate_legend_path is not used ===
    if separate_legend_path is None:
        legend_elements = [Line2D([0], [0], color="gray", lw=10, label="Target (RNA)")]

        if plot_predicted:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="gray",
                    markeredgecolor="black",
                    markersize=10,
                    label="Predicted (AI)",
                )
            )

        leg1 = ax.legend(
            handles=legend_elements, loc="upper left", frameon=False, fontsize=legend_fs
        )
        ax.add_artist(leg1)

    ax.set_title(
        title,
        fontsize=title_fs,
        pad=10,
    )

    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=dpi, format="pdf")
    plt.close(fig)

    # === Optional: export separate legend ===
    if separate_legend_path is not None:
        fig_leg = plt.figure(figsize=(2.5, 0.6))
        ax_leg = fig_leg.add_subplot(111)
        legend_classes = [
            Line2D(
                [0],
                [0],
                marker="s",
                color=colors[j],
                label=classes[j],
                markersize=12,
                linestyle="None",
            )
            for j in range(len(classes))
        ]
        leg = ax_leg.legend(
            handles=legend_classes,
            loc="center",
            ncol=len(classes),
            frameon=True,
            fontsize=legend_fs,
            title="Classes",
            title_fontsize=legend_title_fs,
        )
        ax_leg.axis("off")
        fig_leg.tight_layout()
        fig_leg.savefig(separate_legend_path, dpi=dpi, format="pdf")
        plt.close(fig_leg)


def plot_roc(
    logits,
    targets,
    classes,
    title=None,
    out_path=None,
    legend_pos="bottom right",
    legend_bbox_to_anchor=None,
    font_size=8,
    legend_font_size=6,
    size=(7, 7),
    ax=None,
):

    y_true = targets
    y_score = logits

    owns_ax = ax is None
    if owns_ax:
        fig, ax = plt.subplots(figsize=size)

    for k, cls in enumerate(classes):
        y_true_bin = (y_true == k).astype(int)
        y_score_k = y_score[:, k]

        fpr, tpr, _ = roc_curve(y_true_bin, y_score_k)
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            linewidth=1,
            label=f"{cls} (AUC={roc_auc:.2f})",
        )

    # Random baseline
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")

    ax.set_xlabel(
        "False Positive Rate",
        # fontsize=font_size
    )
    ax.set_ylabel(
        "True Positive Rate",
        # fontsize=font_size
    )

    if title is not None:
        ax.set_title(
            title,
            # fontsize=font_size,
            pad=10,
        )

    ax.tick_params(
        # labelsize=font_size
    )
    ax.grid(True, linestyle="--", alpha=0.5)

    for spine in ax.spines.values():
        spine.set_linewidth(1)

    ax.legend(
        # fontsize=legend_font_size,
        bbox_to_anchor=legend_bbox_to_anchor,
        loc=legend_pos,
        frameon=True,
    )

    return ax


from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt


def plot_ppv(
    logits,
    targets,
    classes,
    title,
    out_path,
    font_size=8,
    legend_font_size=6,
    size=(7, 7),
    show_baseline=True,
):
    y_true = targets  # (n_samples,)
    y_score = logits  # (n_samples, n_classes)

    plt.figure(figsize=size)

    aps = []
    for k in range(len(classes)):
        y_true_bin = (y_true == k).astype(int)
        y_score_k = y_score[:, k]

        ppv, sensitivity, _ = precision_recall_curve(y_true_bin, y_score_k)
        ap = average_precision_score(y_true_bin, y_score_k)
        aps.append(ap)
        # PR plots recall on x, precision on y
        (line,) = plt.plot(
            sensitivity, ppv, linewidth=1, label=f"{classes[k]}"
        )  # (AP = {ap:.2f})")

        if show_baseline:
            prev = y_true_bin.mean()  # class prevalence

            plt.plot(
                [0, 1],
                [prev, prev],
                linestyle="--",
                linewidth=1.5,
                color=line.get_color(),
                label=f"No-skill ({classes[k]})",
            )

    # Axis labels and title
    plt.xlabel("Sensitivity (TPR)", fontsize=font_size)
    plt.ylabel("Predictive Positive Value (PPV)", fontsize=font_size)
    plt.title(f"{title}", fontsize=font_size, pad=15)

    # Ticks
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)

    # Grid and spines
    plt.grid(True, linestyle="--", alpha=0.5)
    for spine in plt.gca().spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    # Legend
    plt.legend(
        fontsize=legend_font_size,
        title_fontsize=legend_font_size,
        loc="lower left",
        frameon=True,
    )

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_npv(
    logits,
    targets,
    classes,
    title="",
    out_path=None,
    font_size=8,
    legend_font_size=6,
    size=(7, 7),
    show_baseline=True,  # baseline = prevalence of negatives
):
    y_true = targets  # (n_samples,)
    y_score = logits  # (n_samples, n_classes)

    plt.figure(figsize=size)

    for k, cls in enumerate(classes):
        # One-vs-rest: positives = class k; for NPV we flip so "negative" is the positive class
        y_pos = (y_true == k).astype(int)
        y_neg = 1 - y_pos  # {0,1}, where 1 means "truly NOT class k"
        s_k = y_score[:, k]
        s_neg = -s_k  # higher score => more negative

        # precision_recall_curve on the flipped task:
        # precision -> NPV, recall -> Specificity (TNR)
        npv, specificity, thr = precision_recall_curve(y_neg, s_neg)
        ap_neg = average_precision_score(y_neg, s_neg)

        (line,) = plt.plot(
            specificity, npv, linewidth=1, label=f"{cls}"
        )  # (AP_neg = {ap_neg:.2f})")

        if show_baseline:
            prev_neg = y_neg.mean()  # prevalence of negatives for class k

            plt.plot(
                [0, 1],
                [prev_neg, prev_neg],
                linestyle="--",
                linewidth=1.5,
                color=line.get_color(),
                label=f"No-skill ({cls})",
            )

    # Axes and cosmetics
    plt.xlabel("Specificity (TNR)", fontsize=font_size)
    plt.ylabel("Negative Predictive Value (NPV)", fontsize=font_size)
    plt.title(f"{title}", fontsize=font_size, pad=15)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.grid(True, linestyle="--", alpha=0.5)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)

    plt.legend(fontsize=legend_font_size, loc="lower right", frameon=True)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300)
    plt.show()


def _check_coord(polys, coord):
    for poly in polys:
        if poly.contains(Point(coord)):
            return True
    return False


def draw_attention_map(
    sample_name,
    sample_attention,
    feats_root,
    images_roots,
    tile_size,
    overlap,
    rois_root=None,
    ds_level=6,
    overlay_blend_alpha=0.4,
    whole_slide=False,
    write_index=False,
    white_background=False,
    write_overlay_annotations=False,
    blur_ksize=None,
    blur_sigma=None,
):
    print("Reading coords...")
    with h5py.File(feats_root / f"{sample_name}.h5", "r") as f:
        coords = f["coords"][()]

    print("Reading tiles...")
    done = False
    for root_folder in images_roots:
        try:
            file = root_folder / f"{sample_name}.jpg"
            print(f"checking {file}")
            image = Image.open(file)
            done = True
            print("Found!")
        except FileNotFoundError as e:
            continue
    if not done:
        print("Couldnt find file")

    print("Downsampling...")
    # Compute the new size
    if ds_level > 0:
        new_width = image.width // (ds_level + 1)
        new_height = image.height // (ds_level + 1)
        new_size = (new_width, new_height)

        # Downsample using high-quality resampling
        image = image.resize(new_size, resample=Image.LANCZOS)

    print("Drawing grid...")

    results = []
    ids = []
    if rois_root:
        print("Selecting ROIs...")
        rois = list(rois_root.glob(sample_name + "*"))
        print(f"Found {len(rois)} ROIs...")

        for r, roi in enumerate(rois, start=1):
            if roi.stem == sample_name:
                ids.append(roi.stem + f"_{r}")
            else:
                ids.append(roi.stem)

            # Read json annotation (from QuPath)
            with open(roi, "r") as handle:
                annots = json.load(handle)["features"]

            print(f"\tROI #{r}")
            roi_polys = [
                Polygon(np.array(annot["geometry"]["coordinates"][0]))
                for annot in annots
            ]

            print("\tFiltering coordinates")
            with ProcessPoolExecutor(max_workers=8) as executor:
                it = executor.map(_check_coord, repeat(roi_polys), coords)
                roi_coords_mask = np.fromiter(
                    tqdm(it, total=len(coords)), dtype=bool, count=len(coords)
                )

            print(f"\tRetained {sum(roi_coords_mask)} coordinates...")
            roi_coords = np.array(coords)[roi_coords_mask]
            roi_attention = sample_attention[roi_coords_mask]

            print("\tDrawing heatmap")
            res = draw_grid_overlaps(
                image,
                coords=roi_coords,
                attention=roi_attention,
                size=(tile_size, tile_size),
                overlap=overlap,
                ds_level=ds_level,
                alpha=overlay_blend_alpha,
                write_index=write_index,
                white_background=white_background,
                write_overlay_annotations=write_overlay_annotations,
                blur_ksize=63,
                blur_sigma=0,
            )

            results.append(res)

    if whole_slide:
        print("Selecting Whole Slide...")
        ids.append(sample_name)
        res = draw_grid_overlaps(
            image,
            coords,
            attention=sample_attention,
            size=(tile_size, tile_size),
            overlap=overlap,
            ds_level=ds_level,
            alpha=overlay_blend_alpha,
            write_index=write_index,
            white_background=white_background,
            write_overlay_annotations=write_overlay_annotations,
            blur_ksize=63,
            blur_sigma=0,
        )
        results.append(res)

    heatmaps = []
    annotations = []
    if write_overlay_annotations:
        for res in results:
            heatmap, annotation = res
            heatmaps.append(heatmap)
            annotations.append(annotation)
    else:
        heatmaps = res

    return heatmaps, ids, annotations


def draw_grid(
    image,
    coords,
    colors,
    size,
    ds_level=6,
    alpha=0.4,
    write_index=False,
    white_background=False,
    blur_ksize=None,
    blur_sigma=None,
):
    if isinstance(image, Image.Image):
        image = np.array(image)

    ds = 2**ds_level
    downsampled_size = (size[0] // ds, size[1] // ds)
    coords = [(x // ds, y // ds) for x, y in coords]

    if white_background:
        base = np.full_like(image, 255)
    else:
        base = image.copy()

    overlay = np.zeros_like(image)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for i, ((x, y), color) in tqdm(
        enumerate(zip(coords, colors)),
        total=len(coords),
        desc="Drawing heat tiles",
    ):
        pt1 = (x, y)
        pt2 = (x + downsampled_size[0], y + downsampled_size[1])
        color = tuple(np.asarray(color, dtype=int).tolist())
        cv2.rectangle(overlay, pt1, pt2, color, -1)
        cv2.rectangle(mask, pt1, pt2, 255, -1)

        if write_index:
            cv2.putText(
                overlay,
                str(i),
                (x + 2, y + 12),
                font=cv2.FONT_HERSHEY_SIMPLEX,
                font_scale=0.5,
                text_color=(0, 255, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    if blur_ksize:
        overlay = cv2.GaussianBlur(overlay, (blur_ksize, blur_ksize), blur_sigma)

    assert overlay.shape[-1] == 3

    # Blend only in masked area
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    base[mask == 255] = blended[mask == 255]

    return base


def draw_grid_overlaps(
    image,
    coords,
    attention,
    size,
    overlap,
    ds_level=6,
    alpha=0.4,
    write_index=False,
    white_background=False,
    write_overlay_annotations=False,
    blur_ksize=None,
    blur_sigma=0,
):
    if isinstance(image, Image.Image):
        image = np.array(image)

    ds = 2**ds_level
    downsampled_size = (size[0] // ds, size[1] // ds)
    coords = [(x // ds, y // ds) for x, y in coords]

    if white_background:
        base = np.full_like(image, 255)
    else:
        base = image.copy()

    overlay = np.zeros(image.shape[:2])
    overlay_count = np.zeros(image.shape[:2])
    mask = np.zeros(image.shape[:2], dtype=bool)

    print("Ranking attention scores...")
    attention = rankdata(attention, "average") / len(attention)
    print("Mapping attention scores...")
    for (x, y), attn in tqdm(
        zip(coords, attention),
        total=len(coords),
        leave=False,
    ):
        pt1 = (x, y)
        pt2 = (x + downsampled_size[0], y + downsampled_size[1])

        overlay[pt1[1] : pt2[1], pt1[0] : pt2[0]] += attn
        overlay_count[pt1[1] : pt2[1], pt1[0] : pt2[0]] += 1

        mask[pt1[1] : pt2[1], pt1[0] : pt2[0]] = 1

    zero_mask = overlay_count == 0
    overlay[~zero_mask] = np.divide(overlay[~zero_mask], overlay_count[~zero_mask])

    if blur_ksize:
        print("Smoothing overlay...")
        overlay_blur = cv2.GaussianBlur(overlay, (blur_ksize, blur_ksize), blur_sigma)

    print("Coloring overlay...")
    cmap = plt.get_cmap("coolwarm")
    overlay_blur = (cmap(overlay_blur)[:, :, :-1] * 255).astype(np.uint8)

    print("Blending overlay...")
    # Blend only in masked area
    blended = cv2.addWeighted(overlay_blur, alpha, image, 1 - alpha, 0)

    print("Masking non-tiled regions...")
    base[mask == 1] = blended[mask == 1]

    if write_index:
        print("Printing tiles indices on image...")
        for i, (x, y) in tqdm(
            enumerate(coords),
            total=len(coords),
            leave=False,
        ):
            cv2.putText(
                img=base,
                text=str(i),
                org=(x + 2, y + 12),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(0, 255, 0),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    if write_overlay_annotations:
        print("Reconstructing attention...")
        new_coords, new_attn = get_attention_from_overlay(
            overlay,
            coords,
            size[0],
            size[0] - overlap,
        )

        print("Drawing on annotation...")
        annotation = convert_attentions_to_annotation(
            new_attn,
            new_coords,
            tile_size=size[0] - overlap,
        )
        return base, annotation

    return base


def kaplan_meier(metadata, time_col, event_col, group_col, weight_col=None):
    groups = metadata[group_col].unique()
    kmf_models = []
    fig, ax = plt.subplots(figsize=(10, 8))

    for group in sorted(groups):
        ix = metadata[group_col] == group
        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=metadata.loc[ix, time_col],
            event_observed=metadata.loc[ix, event_col],
            weights=metadata.loc[ix, weight_col] if weight_col else None,
            label=str(group),
        )
        kmf.plot(ax=ax, ci_show=False, show_censors=True)
        kmf_models.append(kmf)

    # Log-rank test
    results = multivariate_logrank_test(
        metadata[time_col].values,
        metadata[group_col].values,
        metadata[event_col].values,
        metadata[weight_col].values if weight_col else None,
    )
    p = results.p_value

    # Title and labels
    ax.set_title(f"Kaplan-Meier Curve (p = {p:.4g})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.legend(title="Group")

    # Add at-risk table
    add_at_risk_counts(*kmf_models, ax=ax)

    plt.tight_layout()
    plt.show()


def crop_background(img, background_val=0):
    if np.isnan(background_val):
        mask = ~np.isnan(img)
    else:
        mask = img != background_val

    nonzero_rows = np.where(mask.any(axis=1))[0]
    nonzero_cols = np.where(mask.any(axis=0))[0]

    if len(nonzero_rows) == 0 or len(nonzero_cols) == 0:
        raise ValueError("No non-background content found.")

    ymin, ymax = nonzero_rows[0], nonzero_rows[-1] + 1
    xmin, xmax = nonzero_cols[0], nonzero_cols[-1] + 1

    cropped = img[ymin:ymax, xmin:xmax]
    return cropped


def draw_slide(map_slide, us=1, continuous=False, cmap_name=None, normalize=True):
    if continuous:
        cmap = plt.get_cmap(cmap_name or "coolwarm")
        normed = map_slide.astype(np.float32)

        if normalize:
            normed = (normed - np.nanmin(normed)) / (
                np.nanmax(normed) - np.nanmin(normed) + 1e-8
            )

        colored = cmap(normed)[:, :, :3]  # Drop alpha
        colored = (colored * 255).astype(np.uint8)
    else:
        cmap = plt.get_cmap(cmap_name or "tab20")
        colors = (np.array(cmap.colors) * 255)[:, :3]
        colors = np.vstack([[0, 0, 0], [255, 255, 255], colors]).astype(np.uint8)
        colored = colors[map_slide]

    colored_slide = Image.fromarray(colored)
    colored_slide = colored_slide.resize(
        np.array(colored_slide.size) * us, Image.Resampling.NEAREST
    )

    return colored_slide


def map_tiles(
    adata,
    slide_name,
    color_col,
    tile_size=256,
    upsample=1,
    color=True,
    continuous=False,
    cmap_name=None,
    normalize=True,
):
    print("encoding values...")
    color_val = adata.obs[color_col].values
    adata_slide = adata[adata.obs["slide_id"] == slide_name]
    coords = adata_slide.obsm["coordinates"]
    coords = coords // tile_size

    W, H = (coords.max(axis=0) + 1).astype(int)

    if continuous:
        values = color_val[adata.obs["slide_id"] == slide_name].astype(np.float32)
        map_slide = np.full((H, W), np.nan, dtype=np.float32)
        map_slide[coords[:, 1], coords[:, 0]] = values
    else:
        le = LabelEncoder()
        unique_classes = np.array(sorted(np.unique(color_val)))
        le.classes_ = unique_classes
        color_ids = le.fit_transform(color_val) + 1  # +1 to avoid 0 index
        color_ids = color_ids[adata.obs["slide_id"] == slide_name]
        map_slide = np.zeros((H, W), dtype=color_ids.dtype)
        map_slide[coords[:, 1], coords[:, 0]] = color_ids

    if color:
        if continuous:
            print("cropping background...")
            cropped_map = crop_background(map_slide, background_val=np.nan)

            print("drawing continuous image...")
            slide_img = draw_slide(
                cropped_map,
                us=upsample,
                continuous=True,
                cmap_name=cmap_name,
                normalize=normalize,
            )

            return {
                "map": cropped_map,
                "image": slide_img,
            }
        else:
            max_id = color_ids.max()
            chunks = range(1, max_id + 1, 18)
            slides = {}
            for start in chunks:
                end = min(start + 18, max_id + 1)
                print(f"processing {start} to {end}...")

                temp_slide = np.ones_like(map_slide, dtype=map_slide.dtype)
                temp_slide[map_slide == 0] = 0
                for new_val, orig_val in enumerate(range(start, end), start=2):
                    temp_slide[map_slide == orig_val] = new_val

                print("cropping background...")
                temp_slide = crop_background(temp_slide, background_val=0)

                print("drawing tiles...")
                temp_slide = draw_slide(temp_slide, us=upsample)

                classes = np.arange(start, end) - 1
                key = f"{start-1}-{end-1}_map"
                slides[key] = {
                    "class_map": dict(zip(classes, le.inverse_transform(classes))),
                    "image": temp_slide,
                }

            if len(chunks) == 1:
                return slides[list(slides.keys())[0]]

            return slides
    else:
        return map_slide


def boxplots(
    df,
    col="metric",
    row=None,
    sharex=False,
    sharey=False,
    col_wrap=2,
    x="subtype",
    y="score",
    hue="subtype",
    palette="tab10",
    fliersize=0,
    dodge=False,
    jitter=True,
    alpha=0.5,
    linewidth=0.5,
    edgecolor="auto",
    dot_size=3,
    font_size=10,
    xlabel="Subtype",
    ylabel="Score",
    title_clean=True,
):
    g = sns.FacetGrid(
        df,
        col=col,
        row=row,
        sharex=sharex,
        sharey=sharey,
        col_wrap=col_wrap,
    )

    g.map_dataframe(
        sns.boxplot,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        fliersize=fliersize,
        dodge=dodge,
        legend=False,
    )

    g.map_dataframe(
        sns.stripplot,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        dodge=dodge,
        jitter=jitter,
        alpha=alpha,
        linewidth=linewidth,
        edgecolor=edgecolor,
        size=dot_size,
        legend=False,
    )

    # Remove default seaborn titles like "metric = Accuracy"
    if title_clean:
        for ax, title in zip(g.axes.flat, g.col_names):
            clean_title = title.split("=")[-1].strip()
            ax.set_title(clean_title, fontsize=font_size)
    else:
        g.set_titles(col_template="{col_name}", row_template="{row_name}")

    g.set_axis_labels(xlabel, ylabel)
    plt.tight_layout()
    return g


def _read_tiles_for_slide_from_coords(
    sample_id: str,
    coords: List[Tuple[int, int]],
    slide_root: Callable[[str], str],
    level: int,
    tile_size: int,
) -> List[Tuple[str, Image.Image]]:
    slide_path = slide_root / (str(sample_id) + ".ndpi")
    slide = openslide.OpenSlide(slide_path)

    tiles = []
    for x, y in coords:
        tile = slide.read_region(
            (x, y), level=level, size=(tile_size, tile_size)
        ).convert("RGB")
        tiles.append((sample_id, tile))

    slide.close()
    return tiles


def extract_tiles_from_anndata_obs(
    adata: AnnData,
    sample_id_col: str,
    coordinates_key: str,
    level: int,
    tile_size: int,
    slide_root: str,
    num_workers: int = 4,
) -> List[Tuple[str, Image.Image]]:
    """
    Extract tiles using .obsm[coordinates_key] for coordinates.
    Parallelized by slide.
    """
    # Get coordinates from obsm
    coords_matrix = adata.obsm[coordinates_key]
    obs_slice = adata.obs
    obs_names = obs_slice.index

    # Mapping: obs_name → (x, y)
    coords_lookup = {
        name: tuple(coords_matrix[adata.obs_names.get_loc(name)]) for name in obs_names
    }

    # Sample ID → List[(x, y)]
    grouped_coords = obs_slice.groupby(sample_id_col).apply(
        lambda df: [coords_lookup[name] for name in df.index]
    )

    jobs = [(sample_id, coords) for sample_id, coords in grouped_coords.items()]

    # Parallelized reading
    worker_fn = partial(
        _read_tiles_for_slide_from_coords,
        slide_root=slide_root,
        level=level,
        tile_size=tile_size,
    )

    with Pool(processes=num_workers) as pool:
        results = pool.starmap(worker_fn, jobs)

    # Flatten list of lists
    all_tiles = [tile for slide_tiles in results for tile in slide_tiles]
    return all_tiles


import math
import torch
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
from typing import List, Union, Tuple


def make_tile_grid(
    tiles: Union[List[Image.Image], List[Tuple[str, Image.Image]]],
    tile_size: int = 256,
    padding: int = 2,
    pad_value: int = 255,
) -> Image.Image:
    """
    Arrange a perfect square number of tiles into a square grid using torchvision.

    Args:
        tiles: Either a list of PIL.Image.Image or (sample_id, Image) tuples.
        tile_size: size of each tile (assumes all tiles are the same size).
        padding: padding (in pixels) between tiles.
        pad_value: background fill value (0–255) for padding.

    Returns:
        A PIL.Image.Image representing the full grid.
    """
    # Drop sample_id if needed
    if isinstance(tiles[0], tuple):
        tiles = [tile for _, tile in tiles]

    n_tiles = len(tiles)
    sqrt_n = int(math.sqrt(n_tiles))

    if sqrt_n * sqrt_n != n_tiles:
        raise ValueError(f"Number of tiles ({n_tiles}) is not a perfect square.")

    # Convert all to tensors (C, H, W)
    tile_tensors = [
        torch.from_numpy(np.array(tile).transpose(2, 0, 1)).float() / 255.0
        for tile in tiles
    ]

    # Stack into a grid
    grid_tensor = make_grid(
        tile_tensors,
        nrow=sqrt_n,
        padding=padding,
        pad_value=pad_value / 255.0,  # normalize
    )

    # Convert back to PIL
    grid_np = (grid_tensor.numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(grid_np)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def radar_plot_two_rows(
    df: pd.DataFrame,
    labels=None,
    colors=None,
    title=None,
    rmin=0.0,
    rmax=1.0,  # keep radial axis normalized [0,1]
    data_domain=(-1.0, 1.0),  # map data in [-1,1] -> [rmin,rmax]
    rticks_data=None,  # tick labels in data units (e.g., [-1, 0, 1])
    fill_alpha=0.15,
    ax=None,
    mark_max=True,
    dot_size=20,
    dot_edgecolor="white",
    dot_edgewidth=0.8,
    clip=True,  # clip values outside data_domain
    font_size=None,  # <<< NEW: global font size for ticks/title/legend; None = defaults
    legend=True,
):
    """
    Radar chart for exactly 2 rows over N numeric columns.
    Maps data_domain (default [-1,1]) to radial [rmin,rmax] so that
    -1 is at the center, 0 mid-radius, 1 at the circumference.
    """
    # Handle labels from a 'group' column if present
    df = df.copy()
    if "group" in df.columns and labels is None:
        labels = df["group"].astype(str).tolist()
        df = df.drop(columns=["group"])

    # Numeric only & exactly 2 rows
    df_num = df.select_dtypes(include="number")
    if df_num.shape[1] == 0:
        raise ValueError("No numeric columns to plot.")
    if df_num.shape[0] != 2:
        raise ValueError(f"Expected exactly 2 rows, got {df_num.shape[0]}.")

    if labels is None:
        labels = [str(i) for i in df.index[:2]]

    # Categories & angles
    categories = [str(c) for c in df_num.columns]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # Axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        categories, fontsize=font_size if font_size is not None else None
    )

    # Scaling: data -> radial
    vmin, vmax = data_domain
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Invalid data_domain; must be (min, max) with max>min.")

    def to_r(vals):
        vals = np.asarray(vals, dtype=float)
        if clip:
            vals = np.clip(vals, vmin, vmax)
        return rmin + (vals - vmin) / (vmax - vmin) * (rmax - rmin)

    # Radial limits
    ax.set_ylim(rmin, rmax)

    # Radial ticks (labels shown in data units)
    if rticks_data is None:
        rticks_data = np.linspace(vmin, vmax, 5)  # e.g., [-1, -0.5, 0, 0.5, 1]
    rticks_r = to_r(rticks_data)
    ax.set_rlabel_position(0)
    ax.set_yticks(rticks_r)
    ax.set_yticklabels(
        [f"{t:g}" for t in rticks_data],
        fontsize=font_size if font_size is not None else None,
    )

    # Colors
    if colors is None:
        colors = [None, None]

    # Plot the two rows
    for row_values, label, color in zip(df_num.itertuples(index=False), labels, colors):
        vals = list(row_values)
        vals_closed = vals + vals[:1]

        r_vals = to_r(vals_closed)
        (line,) = ax.plot(
            angles, r_vals, linewidth=2, linestyle="solid", label=label, color=color
        )
        used_color = line.get_color()
        ax.fill(angles, r_vals, alpha=fill_alpha, color=used_color)

        # Mark maximum (in data domain)
        if mark_max:
            row_arr = np.asarray(row_values, dtype=float)
            if clip:
                row_arr = np.clip(row_arr, vmin, vmax)
            max_idx = int(np.nanargmax(row_arr))
            ax.scatter(
                [angles[max_idx]],
                [to_r(row_arr[max_idx])],
                s=dot_size,
                color=used_color,
                edgecolors=dot_edgecolor,
                linewidths=dot_edgewidth,
                zorder=5,
            )

    if title:
        if font_size is not None:
            ax.set_title(title, pad=20, fontsize=font_size)
        else:
            ax.set_title(title, pad=20)

    if legend:
        # Legend (respect default styling unless font_size is set)
        legend_kwargs = dict(loc="upper right", bbox_to_anchor=(1.5, 1.1))
        if font_size is not None:
            legend_kwargs["fontsize"] = font_size
        ax.legend(**legend_kwargs)

    # Also ensure tick_params reflect font size if provided
    if font_size is not None:
        ax.tick_params(axis="x", labelsize=font_size)
        ax.tick_params(axis="y", labelsize=font_size)

    return ax


def convert_attentions_to_annotation(
    attention,
    coords,
    colormap="coolwarm",
    color_bins=None,
    tile_size=512,
):

    cmap = plt.get_cmap(colormap)
    # heat_attn = (cmap(attention)[:, :-1] * 255).astype(int)
    print(attention.shape)
    rgbs = (cmap(attention)[:, :-1] * 255).astype(np.uint8)
    print(rgbs.shape)
    # quantilization
    # nbins = 5
    # edges = np.quantile(sample_attn_scaled, np.linspace(0, 1, nbins + 1))
    # levels = np.digitize(sample_attn_scaled, edges[1:-1], right=False)  # 0..nbins-1
    # levels = np.clip(levels, 0, nbins - 1)

    # binning
    # bins = np.linspace(0, 1, nbins + 1)  # 0-0.2, 0.2-0.4, ...
    # levels = np.digitize(attention, bins[1:-1], right=True)  # 0..5
    # levels_colors = (cmap(np.linspace(0, 1, nbins))[:, :-1] * 255).astype(np.uint8)

    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "bbox",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [int(np.min(coords[:, 0])), int(np.min(coords[:, 1]))],
                            [
                                int(np.min(coords[:, 0])),
                                int(np.max(coords[:, 1])) + tile_size,
                            ],
                            [
                                int(np.max(coords[:, 0])) + tile_size,
                                int(np.max(coords[:, 1])) + tile_size,
                            ],
                            [
                                int(np.max(coords[:, 0])) + tile_size,
                                int(np.min(coords[:, 1])),
                            ],
                            [int(np.min(coords[:, 0])), int(np.min(coords[:, 1]))],
                        ]
                    ],
                },
                "properties": {
                    "objectType": "annotation",
                    "classification": {
                        "name": "BoundingBox",
                        "color": [200, 200, 200],  # light gray
                    },
                    "isLocked": True,
                },
            },
        ],
    }

    # for coord, rgb, a, level in zip(coords, heat_attn, attention, levels):
    for coord, rgb, a in zip(coords, rgbs, attention):
        x, y = map(int, coord)

        annotation = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [x, y],
                        [x, y + tile_size],
                        [x + tile_size, y + tile_size],
                        [x + tile_size, y],
                        [x, y],
                    ]
                ],
            },
            "properties": {
                "objectType": "annotation",
                "classification": {
                    "name": "_".join(rgb.astype(str).tolist()),
                    "color": rgb.tolist(),
                    # "name": f"level_{level}",
                    # "color": level_colors[level].tolist(),
                },
                "attention": float(a),
            },
        }
        data["features"].append(annotation)

    return data


def get_attention_from_overlay(overlay, coords, tile_size, stride):
    coords_expanded = []
    attn_expanded = []
    print("Extracting attention...")
    for coord in tqdm(coords, total=len(coords), leave=False):
        for offset in np.arange(0, tile_size, stride):
            offset_coord = coord + offset
            attn_sub_tile = overlay[
                offset_coord[1] : offset_coord[1] + stride,
                offset_coord[0] : offset_coord[0] + stride,
            ]
            if attn_sub_tile.size == 0:
                continue

            assert (attn_sub_tile == attn_sub_tile[0][0]).all(), "Error in slicing"

            coords_expanded.append(offset_coord)
            attn_expanded.append(attn_sub_tile[0][0])

    return np.array(coords_expanded), np.array(attn_expanded)


def plot_corr_bars(
    corrs,
    prefix,
    model_names,
    ref_model,
    palette=None,
    y_axis_name="features",
    title="",
    figsize=(4, 10),
    rows_per_page=20,
    axes_per_row=3,
    pvalues=None,
):
    idx = pd.IndexSlice

    # prepare dataframe
    df = (
        corrs.loc[idx["test", :], :]
        .droplevel(0)
        .T.reset_index(names=y_axis_name)
        .melt(
            id_vars=y_axis_name,
            value_vars=model_names,
            var_name="model",
            value_name="value",
        )
    )

    # rename features
    df[y_axis_name] = (
        df[y_axis_name].str.replace(prefix, "").str.replace("_", " ").str.strip()
    )

    # sort dataframe by values
    levels = (
        df[df["model"] == ref_model]
        .sort_values("value", ascending=False)[y_axis_name]
        .tolist()
    )

    df[y_axis_name] = pd.Categorical(
        df[y_axis_name],
        categories=levels,
        ordered=True,
    )

    df = df.sort_values(y_axis_name)

    # paginate dataframe
    n_models = len(df["model"].unique())
    pages = list(df.groupby(np.arange(len(df)) // (rows_per_page * n_models)))
    n_pages = len(pages)
    if n_pages == 0:
        raise ValueError("No groups to plot")

    # compute number of rows and columns
    rows = max(1, int(np.ceil(n_pages / axes_per_row)))
    columns = int(np.ceil(n_pages / rows))

    # create figure
    figsize = (int(figsize[0] * columns), (figsize[1] * rows))
    fig, axs = plt.subplots(rows, columns, figsize=figsize)
    fig.subplots_adjust(wspace=0.8)
    if columns == 1:
        axs = [axs]
    else:
        axs = axs.ravel()

    # remove empty axes
    for i in range(n_pages, len(axs)):
        fig.delaxes(axs[i])

    # draw one group per axes
    for idx, (ax, (id, df)) in enumerate(zip(axs, pages)):
        # pad last group
        if len(df) < rows_per_page * n_models:
            print("len", len(df))
            print("len", np.ceil(len(df) / n_models))
            missing_rows = int(rows_per_page - len(df) / n_models)
            print("missing", missing_rows)

            # pad with empty data (features will be hidden from yticks downstream)
            cats = df[y_axis_name].cat.categories.union(
                set([f"_EMPTY_{i}" for i in range(missing_rows) for _ in model_names])
            )
            empty_rows = pd.DataFrame(
                {
                    y_axis_name: pd.Categorical(
                        [
                            f"_EMPTY_{i}"
                            for i in range(missing_rows)
                            for _ in range(n_models)
                        ],
                        categories=cats,
                        ordered=True,
                    ),
                    "model": model_names * missing_rows,
                    "value": [0] * (missing_rows * len(model_names)),
                }
            )
            print("empty leb", len(empty_rows))
            df = pd.concat([df, empty_rows])

            df[y_axis_name] = df[y_axis_name].astype("category")
            df[y_axis_name] = df[y_axis_name].cat.set_categories(cats)
            print("len aftr", len(df))
        df[y_axis_name] = df[y_axis_name].cat.remove_unused_categories()

        # plot
        sns.barplot(
            df,
            x="value",
            y=y_axis_name,
            hue="model",
            palette=palette,
            errorbar=None,
            ax=ax,
        )

        # axis manipulation
        ax.set_xticks(np.arange(0, 1, 0.1))
        ax.set_xlim(0, 1)

        # trigger draw to access ticks
        fig.canvas.draw()
        yticks, ylabels = zip(
            *[
                (tick, label)
                for tick, label in zip(ax.get_yticks(), ax.get_yticklabels())
                if not label.get_text().startswith("_EMPTY_")
            ]
        )

        ax.set_yticks(yticks)
        ax.set_yticklabels([lbl.get_text() for lbl in ylabels])

        # draw grid under bars
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right")
    return fig


def fold_scatter_plot(filtered_res, preds_hallmarks, toks=2):
    s = int(np.ceil(np.sqrt(len(filtered_res))))
    fig, axs = plt.subplots(s, s, figsize=(20, 20))
    axs = axs.ravel()

    for i, hallmark_name in enumerate(list(filtered_res.keys())):
        for stage in ["train", "test"]:

            y_true = preds_hallmarks[hallmark_name][f"y_true_{stage}"]
            y_pred = preds_hallmarks[hallmark_name][f"y_pred_{stage}"]

            for k, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
                sns.scatterplot(
                    pd.DataFrame(
                        {
                            "y_true": y_t,
                            "y_pred": y_p,
                        }
                    ),
                    x="y_true",
                    y="y_pred",
                    alpha=1 if stage == "test" else 0.3,
                    label=f"{stage} fold {k}",
                    ax=axs[i],
                    s=12,
                )
                # corrcoef = spearmanr(y_t, y_p).statistic
        corr_test = np.round(filtered_res[hallmark_name]["test"], 2)
        corr_train = np.round(filtered_res[hallmark_name]["train"], 2)

        axs[i].set_title(
            f"{' '.join(hallmark_name.split('_')[toks:])}\ntrain: {corr_train} -> test: {corr_test}"
        )
        axs[i].legend_.remove()

    [ax.remove() for ax in axs[i:]]

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.1, 0.5),
        title="Legend",
    )

    # plt.tight_layout()
    return fig


def plot_shap_beeswarms(results):
    figs = []
    for features_cols in results.keys():
        valid_items = [
            (feat_name, res)
            for feat_name, res in results[features_cols]["shaps"].items()
            if res is not None
        ]

        side = int(np.ceil(np.sqrt(len(valid_items))))

        fig, axs = plt.subplots(side, side, figsize=(30, 30))
        axs = axs.ravel()

        for i, (feat_name, res) in enumerate(valid_items):

            shap_fold = np.concatenate(res["values"])
            X_test_fold = np.concatenate(res["X_test"])
            exp = shap.Explanation(
                values=shap_fold,
                base_values=0.0,  # see note below
                data=X_test_fold,
                feature_names=[
                    str(j) for j in range(shap_fold.shape[1])
                ],  # list of length n_features
            )
            axs[i].set_title(feat_name, fontsize=10)
            shap.plots.beeswarm(exp, ax=axs[i], plot_size=None, show=False)

        for i in range(len(valid_items), len(axs)):
            fig.delaxes(axs[i])

        figs.append(fig)
    return figs
