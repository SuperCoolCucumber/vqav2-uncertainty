"""Evaluate uncertainty vs. explicit abstention behaviour on unanswerable samples."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

try:
    from lm_polygraph.estimators import Estimator
    from lm_polygraph.utils.estimate_uncertainty import estimate_uncertainty
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "lm-polygraph must be installed to run this script. "
        "Install dependencies with `pip install -r requirements.txt`."
    ) from exc

from scripts.run_uncertainty_eval import (
    DEFAULT_SYSTEM_PROMPT,
    ESTIMATOR_REGISTRY,
    build_input_text,
    determine_device,
    find_first_valid_image,
    resolve_local_image_path,
    load_manifest,
    prepare_model,
    resolve_estimators,
)

LOGGER = logging.getLogger("run_abstention_correlation")

ABSTENTION_REGEX = re.compile(
    r"\b(i\s*(?:do\s*not|don't|do n[o’']t)\s*know)\b", re.IGNORECASE
)


def detect_verbal_abstention(text: Optional[str]) -> int:
    """Return 1 if the model explicitly abstains via the regex, else 0."""

    if not text:
        return 0
    if ABSTENTION_REGEX.search(text):
        return 1
    return 0


def compute_correlations(
    scores: Sequence[float], abstain_labels: Sequence[int]
) -> Dict[str, float]:
    valid_mask = ~np.isnan(scores)
    if not np.any(valid_mask):
        return {"pearson": float("nan"), "spearman": float("nan")}

    filtered_scores = np.asarray(scores)[valid_mask]
    filtered_labels = np.asarray(abstain_labels)[valid_mask]

    if filtered_scores.size < 2:
        return {"pearson": float("nan"), "spearman": float("nan")}

    pearson = float(np.corrcoef(filtered_scores, filtered_labels)[0, 1])
    spearman = float(pd.Series(filtered_scores).corr(pd.Series(filtered_labels), method="spearman"))
    return {"pearson": pearson, "spearman": spearman}


def plot_correlation(
    df: pd.DataFrame,
    *,
    uncertainty_col: str,
    abstain_col: str,
    output_path: Path,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.stripplot(
        data=df,
        x=abstain_col,
        y=uncertainty_col,
        jitter=True,
        alpha=0.6,
        ax=ax,
    )
    ax.set_xlabel("Model abstained (regex detected)")
    ax.set_ylabel(uncertainty_col)
    ax.set_title("Uncertainty vs. explicit abstention")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample unanswerable pairs and correlate uncertainty with explicit abstention."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Path to dataset manifest.")
    parser.add_argument("--output-metrics", type=Path, default=Path("data/abstention_metrics.parquet"))
    parser.add_argument("--output-summary", type=Path, default=Path("data/abstention_summary.json"))
    parser.add_argument("--figure-path", type=Path, default=Path("data/uncertainty_vs_abstention_regex.png"))
    parser.add_argument("--model-name", default="microsoft/kosmos-2-patch14-224")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--answerable-column", default=None)
    parser.add_argument("--unanswerable-value", type=int, default=0, help="Value indicating an unanswerable item (used only if column provided).")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--estimator",
        default="mean_token_entropy",
        help=f"Uncertainty estimator to use. Available: {', '.join(ESTIMATOR_REGISTRY)}",
    )
    parser.add_argument("--system-prompt", nargs="?", default=None, const=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional directory containing the images referenced by the manifest.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    np.random.seed(args.seed)

    manifest = load_manifest(args.manifest)
    if args.answerable_column and args.answerable_column in manifest.columns:
        filtered = manifest[manifest[args.answerable_column] == args.unanswerable_value]
        if filtered.empty:
            LOGGER.warning("No rows matched %s == %s; using entire manifest instead.", args.answerable_column, args.unanswerable_value)
            filtered = manifest
    else:
        filtered = manifest

    sampled = filtered.sample(
        n=min(args.num_samples, len(filtered)),
        random_state=args.seed,
    ).reset_index(drop=True)
    LOGGER.info("Selected %d unanswerable samples for evaluation.", len(sampled))

    if args.question_column not in sampled.columns or args.image_column not in sampled.columns:
        raise KeyError("Manifest missing required question/image columns.")

    estimator_spec = resolve_estimators([args.estimator])[0]
    estimator: Estimator = estimator_spec.factory()

    image_root = args.image_root.resolve() if args.image_root else None

    initial_image_path = find_first_valid_image(sampled[args.image_column], image_root)
    device = determine_device(args.device)
    model = prepare_model(
        args.model_name,
        device,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        initial_image_path=initial_image_path,
        max_new_tokens=args.max_new_tokens,
    )

    system_prompt = args.system_prompt.strip() if args.system_prompt else None

    rows: List[Dict[str, object]] = []
    for row in sampled.itertuples(index=False):
        image_path = getattr(row, args.image_column)
        question = getattr(row, args.question_column)
        answer = getattr(row, args.answer_column, None) if args.answer_column in sampled.columns else None

        resolved_image = resolve_local_image_path(image_path, image_root)
        if resolved_image is None:
            LOGGER.warning("Skipping sample with missing image path: %s", image_path)
            continue

        input_text = build_input_text(str(question), system_prompt)

        with Image.open(resolved_image) as img:
            model.images = [img.convert("RGB").copy()]

        ue_output = estimate_uncertainty(model, estimator, input_text=input_text)
        score = float(np.asarray(ue_output.uncertainty).item())
        generation = ue_output.generation_text
        abstain_flag = detect_verbal_abstention(generation)

        rows.append(
            {
                "question": question,
                "image_path": str(resolved_image),
                "image_reference": image_path,
                "answer": answer,
                "input_text": input_text,
                "model_answer": generation,
                "uncertainty": score,
                "verbal_abstain": abstain_flag,
            }
        )

    if not rows:
        LOGGER.error("No samples processed successfully.")
        return

    result_df = pd.DataFrame(rows)
    result_df.to_parquet(args.output_metrics, index=False)
    LOGGER.info("Saved per-sample metrics to %s", args.output_metrics)

    correlations = compute_correlations(result_df["uncertainty"], result_df["verbal_abstain"])
    abstention_rate = float(result_df["verbal_abstain"].mean())

    summary = {
        "num_samples": int(result_df.shape[0]),
        "estimator": estimator_spec.key,
        "verbal_abstention_rate": abstention_rate,
        "correlations": correlations,
    }

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Wrote summary to %s", args.output_summary)

    try:
        plot_correlation(
            result_df,
            uncertainty_col="uncertainty",
            abstain_col="verbal_abstain",
            output_path=args.figure_path,
        )
        LOGGER.info("Saved correlation figure to %s", args.figure_path)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to generate figure: %s", exc)


if __name__ == "__main__":
    main()
