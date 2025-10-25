"""Analyse correlation between uncertainty scores and binary outcomes."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LOGGER = logging.getLogger("analyse_uncertainty")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute summary statistics for uncertainty vs. abstention/correctness."
    )
    parser.add_argument("--metrics", type=Path, required=True, help="Path to metrics parquet/jsonl/csv file.")
    parser.add_argument("--estimator-key", default="mean_token_entropy", help="Estimator name used in metrics (without the 'uncertainty__' prefix).")
    parser.add_argument("--abstain-column", default=None, help="Optional column with abstention decisions (0/1).")
    parser.add_argument("--label-column", default=None, help="Optional ground-truth column (e.g. correctness flag).")
    parser.add_argument("--figure-out", type=Path, default=Path("data/uncertainty_analysis.png"))
    parser.add_argument("--report-out", type=Path, default=Path("data/uncertainty_analysis.json"))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".jsonl":
        return pd.read_json(path, orient="records", lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported metrics file type: {suffix}")


def compute_correlations(df: pd.DataFrame, x_col: str, y_col: str) -> dict:
    valid = df[[x_col, y_col]].dropna()
    if valid.empty:
        return {"pearson": np.nan, "spearman": np.nan, "kendall": np.nan}
    return {
        "pearson": float(valid[x_col].corr(valid[y_col], method="pearson")),
        "spearman": float(valid[x_col].corr(valid[y_col], method="spearman")),
        "kendall": float(valid[x_col].corr(valid[y_col], method="kendall")),
    }


def plot_relationship(
    df: pd.DataFrame,
    *,
    x_col: str,
    abstain_col: Optional[str],
    label_col: Optional[str],
    output_path: Path,
) -> None:
    plot_columns = []
    if abstain_col and abstain_col in df.columns:
        plot_columns.append(("Abstention", abstain_col))
    if label_col and label_col in df.columns:
        plot_columns.append(("Label", label_col))

    if not plot_columns:
        LOGGER.info("No categorical columns provided for plotting; skipping figure generation.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, len(plot_columns), figsize=(6 * len(plot_columns), 5))
    if len(plot_columns) == 1:
        axes = [axes]

    for ax, (title, column) in zip(axes, plot_columns):
        try:
            sns.kdeplot(
                data=df,
                x=x_col,
                hue=column,
                ax=ax,
                common_norm=False,
                fill=True,
            )
            ax.set_ylabel("Density")
        except Exception:
            sns.stripplot(data=df, x=column, y=x_col, ax=ax, jitter=True, alpha=0.6)
            ax.set_ylabel(x_col)
        ax.set_title(f"Uncertainty vs. {title.lower()}")
        ax.set_xlabel(column)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = load_metrics(args.metrics)
    uncertainty_col = f"uncertainty__{args.estimator_key}"

    if uncertainty_col not in df.columns:
        raise KeyError(
            f"Column '{uncertainty_col}' not found. Available columns: {df.columns.tolist()}"
        )
    if args.abstain_column and args.abstain_column not in df.columns:
        LOGGER.warning(
            "Abstain column '%s' not found; skipping abstention analysis.",
            args.abstain_column,
        )
        args.abstain_column = None

    summary = {
        "num_samples": int(df.shape[0]),
        "estimator": args.estimator_key,
        "abstain_column": args.abstain_column,
    }
    summary["uncertainty_statistics"] = {
        "mean": float(df[uncertainty_col].mean()),
        "std": float(df[uncertainty_col].std(ddof=0)),
        "min": float(df[uncertainty_col].min()),
        "max": float(df[uncertainty_col].max()),
    }
    if args.abstain_column:
        summary["abstention_rate"] = float(df[args.abstain_column].mean())
        summary["uncertainty_vs_abstain"] = compute_correlations(
            df, uncertainty_col, args.abstain_column
        )
    else:
        summary["abstention_rate"] = None

    if args.label_column:
        if args.label_column not in df.columns:
            LOGGER.warning(
                "Requested label column '%s' not present; skipping label analysis.",
                args.label_column,
            )
        else:
            label_df = df[[uncertainty_col, args.label_column]].dropna()
            if not label_df.empty:
                label_df = label_df.copy()
                try:
                    label_df[args.label_column] = pd.to_numeric(
                        label_df[args.label_column], errors="coerce"
                    )
                except Exception:
                    pass
                summary["uncertainty_vs_label"] = compute_correlations(
                    label_df, uncertainty_col, args.label_column
                )

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with args.report_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Saved analysis summary to %s", args.report_out)

    try:
        plot_relationship(
            df,
            x_col=uncertainty_col,
            abstain_col=args.abstain_column,
            label_col=args.label_column,
            output_path=args.figure_out,
        )
        LOGGER.info("Saved analysis figure to %s", args.figure_out)
    except Exception as exc:
        LOGGER.warning("Failed to generate figure: %s", exc)


if __name__ == "__main__":
    main()
