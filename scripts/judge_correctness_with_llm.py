"""Augment uncertainty metrics with LLM-based correctness judgments."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Sequence

import pandas as pd

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Please install the `openai` package to use this script.") from exc


LOGGER = logging.getLogger("judge_correctness")


PROMPT_TEMPLATE = """
You are an expert VQA evaluator. Judge whether the model's answer correctly addresses the question using the list of acceptable answers. Any paraphrase or equivalent meaning should be counted as correct. Output a single token: "CORRECT" or "INCORRECT".

Question: {question}
Model answer: {model_answer}
Acceptable answers: {gold_answers}
""".strip()


def call_judge(client: OpenAI, model: str, question: str, answer: str, gold_answers: List[str]) -> str:
    prompt = PROMPT_TEMPLATE.format(
        question=question,
        model_answer=answer or "",
        gold_answers=", ".join(gold_answers),
    )
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=0,
    )
    text = response.output[0].content[0].text.strip().upper()
    return text


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use OpenAI to judge VQA correctness.")
    parser.add_argument("--metrics", type=Path, default=Path("data/uncertainty_metrics.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/uncertainty_metrics_with_judge.parquet"))
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model ID (Responses API compatible).")
    parser.add_argument("--api-key", default=None, help="Optional override for OPENAI_API_KEY.")
    parser.add_argument("--start", type=int, default=0, help="Optional starting index (resume).")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to judge.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.metrics.exists():
        raise FileNotFoundError(args.metrics)

    df = pd.read_parquet(args.metrics)
    if "gold_answers" not in df.columns or "model_answer" not in df.columns:
        raise KeyError("Metrics file must contain 'gold_answers' and 'model_answer' columns.")

    subset = df.iloc[args.start : args.start + args.limit if args.limit else None].copy()
    LOGGER.info("Judging %d samples using %s", subset.shape[0], args.model)

    api_key = args.api_key
    client = OpenAI(api_key=api_key) if api_key else OpenAI()

    judgements: List[str] = []
    for idx, row in subset.iterrows():
        gold_answers = row["gold_answers"]
        if isinstance(gold_answers, str):
            try:
                gold_answers = json.loads(gold_answers)
            except json.JSONDecodeError:
                gold_answers = [gold_answers]
        if not isinstance(gold_answers, list):
            gold_answers = [str(gold_answers)]
        text = call_judge(
            client,
            args.model,
            question=row.get("question", ""),
            answer=row.get("model_answer", ""),
            gold_answers=[str(ans) for ans in gold_answers],
        )
        judgements.append(text)
    subset["judge_llm_correctness"] = [1 if j.startswith("CORRECT") else 0 for j in judgements]
    df.loc[subset.index, "judge_llm_correctness"] = subset["judge_llm_correctness"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    LOGGER.info("Wrote augmented metrics to %s", args.output)


if __name__ == "__main__":
    main()
