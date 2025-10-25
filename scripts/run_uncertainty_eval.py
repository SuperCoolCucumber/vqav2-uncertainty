"""Run VisualWhiteboxModel uncertainty evaluation on a prepared manifest."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

try:
    from lm_polygraph.estimators import (
        Estimator,
        MeanTokenEntropy,
        MaximumSequenceProbability,
        MonteCarloSequenceEntropy,
        Perplexity,
        SemanticEntropy,
    )
    from lm_polygraph.model_adapters.visual_whitebox_model import VisualWhiteboxModel
    from lm_polygraph.utils.estimate_uncertainty import estimate_uncertainty
    from lm_polygraph.utils.generation_parameters import GenerationParameters
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "lm-polygraph is required for this script. Install it via `pip install -r requirements.txt` "
        "with Python>=3.10."
    ) from exc

from transformers import AutoModelForVision2Seq, AutoProcessor

LOGGER = logging.getLogger("run_uncertainty_eval")

DEFAULT_SYSTEM_PROMPT = (
    'Answer the question about the image. If the image does not contain enough '
    'information to answer, reply with "I don\'t know."'
)


@dataclass(frozen=True)
class EstimatorSpec:
    """Metadata about an uncertainty estimator."""

    key: str
    factory: type[Estimator]
    higher_is_more_uncertain: bool = True


ESTIMATOR_REGISTRY: Dict[str, EstimatorSpec] = {
    "mean_token_entropy": EstimatorSpec(
        key="mean_token_entropy",
        factory=MeanTokenEntropy,
        higher_is_more_uncertain=True,
    ),
    "maximum_sequence_probability": EstimatorSpec(
        key="maximum_sequence_probability",
        factory=MaximumSequenceProbability,
        higher_is_more_uncertain=False,
    ),
    "monte_carlo_sequence_entropy": EstimatorSpec(
        key="monte_carlo_sequence_entropy",
        factory=MonteCarloSequenceEntropy,
        higher_is_more_uncertain=True,
    ),
    "perplexity": EstimatorSpec(
        key="perplexity",
        factory=Perplexity,
        higher_is_more_uncertain=True,
    ),
    "semantic_entropy": EstimatorSpec(
        key="semantic_entropy",
        factory=SemanticEntropy,
        higher_is_more_uncertain=True,
    ),
}


def resolve_estimators(keys: Sequence[str]) -> List[EstimatorSpec]:
    specs: List[EstimatorSpec] = []
    for key in keys:
        lower = key.lower()
        if lower not in ESTIMATOR_REGISTRY:
            raise KeyError(
                f"Estimator '{key}' is not registered. Available: {', '.join(ESTIMATOR_REGISTRY)}"
            )
        specs.append(ESTIMATOR_REGISTRY[lower])
    return specs


def determine_device(explicit: Optional[str] = None) -> torch.device:
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if path.suffix == ".jsonl":
        return pd.read_json(path, orient="records", lines=True)
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported manifest format: {path.suffix}")


def resolve_local_image_path(value: Any, image_root: Optional[Path]) -> Optional[Path]:
    if isinstance(value, str) and value:
        candidate = Path(value)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if image_root is not None:
            combined = image_root / value
            if combined.exists():
                return combined
        if candidate.exists():
            return candidate
    return None


def find_first_valid_image(paths: Iterable[Any], image_root: Optional[Path]) -> Path:
    for value in paths:
        resolved = resolve_local_image_path(value, image_root)
        if resolved is not None:
            return resolved
    raise FileNotFoundError(
        "Unable to find a valid image path in the manifest. Ensure `image_path` column is populated."
    )


def prepare_model(
    model_name: str,
    device: torch.device,
    *,
    trust_remote_code: bool,
    dtype: Optional[str],
    initial_image_path: Path | str,
    max_new_tokens: int,
) -> VisualWhiteboxModel:
    LOGGER.info("Loading model %s on device %s", model_name, device)

    dtype_obj: Optional[torch.dtype] = None
    if dtype:
        try:
            dtype_obj = getattr(torch, dtype)
        except AttributeError as exc:
            raise ValueError(f"Unknown torch dtype '{dtype}'.") from exc

    model_kwargs: Dict[str, Any] = {}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if dtype_obj is not None:
        model_kwargs["torch_dtype"] = dtype_obj

    base_model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)
    base_model.to(device)
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    generation_params = GenerationParameters(max_new_tokens=max_new_tokens)
    vw_model = VisualWhiteboxModel(
        base_model,
        processor,
        model_path=model_name,
        model_type="VisualLM",
        image_paths=[str(initial_image_path)],
        generation_parameters=generation_params,
    )
    vw_model.images = []  # Will be populated per-sample.
    return vw_model


def compute_threshold(
    scores: Sequence[float],
    *,
    higher_is_more_uncertain: bool,
    explicit_threshold: Optional[float],
    target_rate: Optional[float],
) -> Optional[float]:
    if explicit_threshold is not None:
        return explicit_threshold
    if target_rate is None:
        return None
    if not scores:
        return None
    if target_rate <= 0 or target_rate >= 1:
        raise ValueError("target_abstention_rate must be between 0 and 1 (exclusive).")
    array = np.asarray(scores, dtype=np.float64)
    if higher_is_more_uncertain:
        q = 1.0 - target_rate
    else:
        q = target_rate
    return float(np.quantile(array, q))


def apply_abstention_rule(
    scores: Sequence[float],
    *,
    threshold: float,
    higher_is_more_uncertain: bool,
) -> List[int]:
    decisions: List[int] = []
    for score in scores:
        if np.isnan(score):
            decisions.append(0)
        else:
            if higher_is_more_uncertain:
                decisions.append(int(score >= threshold))
            else:
                decisions.append(int(score <= threshold))
    return decisions


def build_input_text(question: str, system_prompt: Optional[str]) -> str:
    """Compose model input text with an optional system prompt."""

    if system_prompt:
        prompt = system_prompt.strip()
        return f"{prompt}\n\nQuestion: {question}"
    return question


def parse_answers_field(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(a) for a in raw]
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [str(a) for a in data]
        except json.JSONDecodeError:
            # treat as semicolon/comma separated string fallback
            parts = re.split(r"[;|]", raw)
            return [p.strip() for p in parts if p.strip()]
        return [raw.strip()]
    return [str(raw)]


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def compute_exact_match(model_answer: Optional[str], gold_answers: List[str]) -> int:
    if not model_answer:
        return 0
    norm_pred = normalize_answer(model_answer)
    normalized_gold = {normalize_answer(ans) for ans in gold_answers if ans}
    return int(norm_pred in normalized_gold)


def compute_score_label_correlations(
    scores: pd.Series, labels: pd.Series
) -> Dict[str, float]:
    valid = pd.concat([scores, labels], axis=1).dropna()
    if len(valid) < 2:
        return {"pearson": float("nan"), "spearman": float("nan")}
    pearson = float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="pearson"))
    spearman = float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman"))
    return {"pearson": pearson, "spearman": spearman}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate VisualWhiteboxModel uncertainty metrics on a dataset manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/uncertainty_metrics.parquet"))
    parser.add_argument("--model-name", default="microsoft/kosmos-2-patch14-224")
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default=None, help="Optional single target answer column.")
    parser.add_argument("--answers-column", default="answers", help="Column containing list/JSON of acceptable answers.")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--multiple-choice-column", default="multiple_choice_answer")
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on processed samples.")
    parser.add_argument(
        "--estimators",
        nargs="+",
        default=["mean_token_entropy"],
        help=f"Estimator names. Available: {', '.join(ESTIMATOR_REGISTRY)}",
    )
    parser.add_argument("--abstention-estimator", default=None, help="Estimator to use for abstention decisions. Defaults to the first estimator.")
    parser.add_argument("--abstention-threshold", type=float, default=None)
    parser.add_argument("--target-abstention-rate", type=float, default=None, help="If provided, pick threshold to achieve this abstention rate (overrides --abstention-threshold).")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None, help="Optional torch dtype, e.g. float16.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--system-prompt",
        nargs="?",
        default=None,
        const=DEFAULT_SYSTEM_PROMPT,
        help=(
            "Optional instruction prefix to prepend to each question. "
            "If provided without a value, uses the default: "
            f'"{DEFAULT_SYSTEM_PROMPT}"'
        ),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional directory to prefix to relative image paths stored in the manifest.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-path", type=Path, default=Path("data/uncertainty_summary.json"))
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest = load_manifest(args.manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit)
        LOGGER.info("Limiting evaluation to %d samples", len(manifest))

    if args.question_column not in manifest.columns:
        raise KeyError(
            f"Question column '{args.question_column}' not present in manifest columns: {manifest.columns.tolist()}"
        )
    if args.image_column not in manifest.columns:
        raise KeyError(
            f"Image column '{args.image_column}' not present: {manifest.columns.tolist()}"
        )

    estimators = resolve_estimators(args.estimators)
    abstention_estimator_key = (args.abstention_estimator or estimators[0].key).lower()
    abstention_spec = next(spec for spec in estimators if spec.key == abstention_estimator_key)

    if args.image_root is not None:
        image_root = args.image_root.resolve()
    else:
        image_root = None

    initial_image_path = find_first_valid_image(manifest[args.image_column], image_root)
    device = determine_device(args.device)
    model = prepare_model(
        args.model_name,
        device,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        initial_image_path=initial_image_path,
        max_new_tokens=args.max_new_tokens,
    )

    instantiated_estimators: Dict[str, Estimator] = {
        spec.key: spec.factory() for spec in estimators
    }

    records: List[Dict[str, Any]] = []
    primary_scores: List[float] = []
    system_prompt = args.system_prompt.strip() if args.system_prompt else None

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Evaluating"):
        image_value = getattr(row, args.image_column)
        resolved_image = resolve_local_image_path(image_value, image_root)
        if resolved_image is None:
            LOGGER.warning("Skipping sample due to missing image path: %s", image_value)
            continue

        question = getattr(row, args.question_column)
        sample_id = getattr(row, "sample_id", None) or getattr(row, "id", None)
        single_answer = None
        if args.answer_column and args.answer_column in manifest.columns:
            single_answer = getattr(row, args.answer_column)
        label = (
            getattr(row, args.label_column)
            if args.label_column and args.label_column in manifest.columns
            else None
        )

        answers_list: List[str] = []
        if args.answers_column and args.answers_column in manifest.columns:
            answers_list = parse_answers_field(getattr(row, args.answers_column))
        if single_answer:
            answers_list.append(str(single_answer))
        if args.multiple_choice_column and args.multiple_choice_column in manifest.columns:
            mc_answer = getattr(row, args.multiple_choice_column)
            if mc_answer:
                answers_list.append(str(mc_answer))
        # Deduplicate preserving order
        seen = set()
        dedup_answers = []
        for ans in answers_list:
            if ans is None:
                continue
            key = normalize_answer(ans)
            if key not in seen:
                dedup_answers.append(ans)
                seen.add(key)
        answers_list = dedup_answers

        input_text = build_input_text(str(question), system_prompt)

        with Image.open(resolved_image) as img:
            model.images = [img.convert("RGB").copy()]

        sample_metrics: Dict[str, Any] = {
            "image_path": str(resolved_image),
            "image_reference": image_value,
            "question": question,
            "answer": single_answer,
            "label": label,
            "input_text": input_text,
            "gold_answers": json.dumps(answers_list, ensure_ascii=False),
        }
        if sample_id is not None:
            sample_metrics["sample_id"] = sample_id

        generation_text: Optional[str] = None

        for key, estimator in instantiated_estimators.items():
            ue_output = estimate_uncertainty(model, estimator, input_text=input_text)
            score = float(np.asarray(ue_output.uncertainty).item())
            sample_metrics[f"uncertainty__{key}"] = score
            sample_metrics[f"estimator__{key}"] = ue_output.estimator
            sample_metrics[f"tokens__{key}"] = json.dumps(ue_output.generation_tokens)
            if generation_text is None:
                generation_text = ue_output.generation_text
            if key == abstention_spec.key:
                primary_scores.append(score)

        sample_metrics["model_answer"] = generation_text
        correct_flag = compute_exact_match(generation_text, answers_list)
        sample_metrics["correct_exact"] = correct_flag
        records.append(sample_metrics)

    if not records:
        LOGGER.error("No samples were processed. Abort.")
        return

    metrics_df = pd.DataFrame.from_records(records)
    if "correct_exact" in metrics_df.columns:
        summary_accuracy = float(metrics_df["correct_exact"].mean())
    else:
        summary_accuracy = None

    threshold = compute_threshold(
        primary_scores,
        higher_is_more_uncertain=abstention_spec.higher_is_more_uncertain,
        explicit_threshold=args.abstention_threshold,
        target_rate=args.target_abstention_rate,
    )

    summary: Dict[str, Any] = {
        "total_samples": len(metrics_df),
        "estimators": list(instantiated_estimators.keys()),
        "primary_estimator": abstention_spec.key,
    }
    if summary_accuracy is not None:
        summary["accuracy_exact_match"] = summary_accuracy

    if threshold is not None:
        decisions = apply_abstention_rule(
            primary_scores,
            threshold=threshold,
            higher_is_more_uncertain=abstention_spec.higher_is_more_uncertain,
        )
        metrics_df["abstain_decision"] = decisions
        metrics_df["abstention_threshold"] = threshold
        abstention_rate = float(np.mean(decisions))
        summary.update(
            {
                "abstention_threshold": threshold,
                "abstention_rate": abstention_rate,
                "abstention_orientation": "higher"
                if abstention_spec.higher_is_more_uncertain
                else "lower",
            }
        )
        LOGGER.info(
            "Primary estimator %s: threshold=%s, abstention_rate=%.3f",
            abstention_spec.key,
            threshold,
            abstention_rate,
        )
    else:
        LOGGER.info("No abstention threshold provided; skipping abstention computation.")

    if "correct_exact" in metrics_df.columns:
        correctness_series = metrics_df["correct_exact"].astype(float)
        for key in instantiated_estimators:
            col = f"uncertainty__{key}"
            if col in metrics_df.columns:
                summary[f"correlation_uncertainty_vs_correct__{key}"] = compute_score_label_correlations(
                    metrics_df[col], correctness_series
                )

    if args.label_column and args.label_column in metrics_df.columns:
        try:
            label_series = pd.to_numeric(metrics_df[args.label_column], errors="coerce")
            if threshold is not None:
                coverage = float(np.mean(1 - metrics_df["abstain_decision"]))
            else:
                coverage = None
            summary["label_coverage_fraction"] = coverage
            summary["label_valid_fraction"] = float(label_series.notna().mean())
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to derive label-based summary: %s", exc)

    summary_path = args.summary_path
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    LOGGER.info("Wrote summary to %s", summary_path)

    metrics_df.to_parquet(args.output, index=False)
    LOGGER.info("Saved per-sample metrics to %s", args.output)


if __name__ == "__main__":
    main()
