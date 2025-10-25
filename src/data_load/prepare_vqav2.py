"""Sample VQAv2 validation examples and build a local manifest."""

from __future__ import annotations

import argparse
import json
import logging
import itertools
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


def sample_vqav2(
    output_dir: Path,
    *,
    images_dir: Path,
    num_samples: int,
    dataset_name: str = "lmms-lab/VQAv2",
    split: str = "validation",
    streaming: bool = True,
) -> pd.DataFrame:
    LOGGER.info(
        "Loading dataset %s split=%s streaming=%s",
        dataset_name,
        split,
        streaming,
    )
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
    )

    if streaming:
        LOGGER.info("Collecting first %d samples from the streaming iterator.", num_samples)
        samples = list(itertools.islice(dataset, num_samples))
    else:
        LOGGER.info("Selecting first %d samples from the dataset.", num_samples)
        subset = dataset.select(range(min(num_samples, len(dataset))))
        samples = list(subset)

    records: List[Dict[str, Any]] = []
    images_dir.mkdir(parents=True, exist_ok=True)

    for sample in tqdm(samples, desc="Saving samples"):
        image: Image.Image = sample["image"]
        question_id = sample["question_id"]
        image_filename = f"vqav2_{question_id}.jpg"
        image_path = images_dir / image_filename
        image.save(image_path, format="JPEG")

        answers = [ans["answer"] for ans in sample.get("answers", []) if "answer" in ans]
        record = {
            "sample_id": str(question_id),
            "question_id": int(question_id),
            "image_id": int(sample["image_id"]),
            "question": sample["question"],
            "answers": json.dumps(answers, ensure_ascii=False),
            "multiple_choice_answer": sample["multiple_choice_answer"],
            "answer_type": sample.get("answer_type"),
            "question_type": sample.get("question_type"),
            "image_path": str(image_path),
        }
        records.append(record)

    frame = pd.DataFrame(records)
    return frame


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a VQAv2 manifest with sampled validation examples.")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--manifest-name", default="vqav2_sample_manifest.parquet")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--streaming", action="store_true", help="Enable streaming mode when loading the dataset.")
    parser.add_argument("--dataset-name", default="lmms-lab/VQAv2")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(args)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = sample_vqav2(
        args.output_dir,
        images_dir=args.images_dir,
        num_samples=args.num_samples,
        dataset_name=args.dataset_name,
        split=args.split,
        streaming=args.streaming,
    )

    manifest_path = args.output_dir / args.manifest_name
    frame.to_parquet(manifest_path, index=False)
    LOGGER.info("Saved manifest with %d rows to %s", len(frame), manifest_path)


if __name__ == "__main__":
    main()
