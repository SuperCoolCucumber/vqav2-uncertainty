"""Utilities for loading the CertainlyUncertain dataset and preparing manifests."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import pandas as pd
from datasets import Dataset, IterableDataset, load_dataset

LOGGER = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    """Lightweight representation of a dataset element for downstream processing."""

    sample_id: str
    question: str
    answer: Optional[str]
    image_path: Optional[str]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "answer": self.answer,
            "image_path": self.image_path,
            "meta": json.dumps(self.meta, ensure_ascii=False),
        }


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _extract_image_path(image_value: Any, output_dir: Optional[Path] = None) -> Optional[str]:
    """Attempt to resolve an image path from a datasets Image feature.

    For remote data, users may need to provide a custom download script; we handle the
    common cases (local cached files, PIL Image objects, dict representation).
    """

    if image_value is None:
        return None

    # Hugging Face Image feature often returns a dict with "path".
    if isinstance(image_value, Mapping):
        if "path" in image_value and image_value["path"]:
            return image_value["path"]
        if "bytes" in image_value and output_dir is not None:
            # Persist raw bytes to disk if requested.
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / f"image_{hash(image_value['bytes']) & 0xFFFFFFFF:x}.png"
            if not target.exists():
                with target.open("wb") as f:
                    f.write(image_value["bytes"])
            return str(target)

    # PIL Image case: save to disk when output dir is available.
    try:
        from PIL import Image  # type: ignore

        if isinstance(image_value, Image.Image) and output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / f"image_{id(image_value):x}.png"
            if not target.exists():
                image_value.save(target)
            return str(target)
    except ImportError:
        pass

    # Fallback: string-like path.
    if isinstance(image_value, (str, Path)):
        return str(image_value)

    return None


def _extract_from_conversations(conversations: Any) -> Tuple[Optional[str], Optional[str]]:
    """Pull the first human question and first model answer from a conversation list."""

    if not isinstance(conversations, Sequence):
        return None, None

    question: Optional[str] = None
    answer: Optional[str] = None
    for turn in conversations:
        if not isinstance(turn, Mapping):
            continue
        role = turn.get("from")
        value = turn.get("value")
        if not isinstance(value, str):
            continue
        if role == "human" and question is None:
            question = value
        elif role == "gpt" and answer is None:
            answer = value
        if question is not None and answer is not None:
            break

    return question, answer


def _clean_question(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    cleaned = text.replace("<image>", "").strip()
    # Remove leading newlines introduced by "<image>\n".
    return cleaned.lstrip("\n").strip()


def _clean_answer(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    return text.strip()


def extract_default_fields(
    example: MutableMapping[str, Any],
    *,
    question_field: str,
    answer_field: Optional[str],
    image_field: Optional[str],
    conversation_field: Optional[str] = None,
    additional_fields: Optional[Sequence[str]] = None,
    asset_output_dir: Optional[Path] = None,
) -> ManifestEntry:
    """Extract standard fields from a datasets example."""

    question: Optional[str] = None
    if question_field:
        q_value = example.get(question_field)
        if q_value is not None:
            question = str(q_value)

    answer: Optional[str] = None
    if answer_field:
        answer = example.get(answer_field)
        if isinstance(answer, list):
            answer = answer[0] if answer else None

    conv_question: Optional[str] = None
    conv_answer: Optional[str] = None
    if (question is None or question == "" or answer is None) and conversation_field:
        conv_question, conv_answer = _extract_from_conversations(example.get(conversation_field))

    if question is None or question == "":
        question = conv_question
    if answer is None:
        answer = conv_answer

    question = _clean_question(question)
    answer = _clean_answer(answer)

    if question is None or question == "":
        raise KeyError(
            f"Unable to locate a question for the sample; checked '{question_field}' and '{conversation_field}'."
        )

    sample_id = example.get("id") or example.get("uid") or example.get("sample_id")
    if sample_id is None:
        sample_id = str(example.get("question_id") or example.get("image_id") or "")
    if not sample_id:
        raise KeyError("Unable to infer a unique sample identifier from the example.")

    image_path: Optional[str] = None
    if image_field:
        image_value = example.get(image_field)
        image_path = _extract_image_path(image_value, asset_output_dir)

    meta: Dict[str, Any] = {}
    if additional_fields:
        for field in additional_fields:
            if field in example:
                meta[field] = example[field]

    return ManifestEntry(
        sample_id=str(sample_id),
        question=str(question),
        answer=str(answer) if answer is not None else None,
        image_path=image_path,
        meta=meta,
    )


def build_dataset_manifest(
    dataset: Iterable[MutableMapping[str, Any]],
    *,
    question_field: str = "question",
    answer_field: Optional[str] = "answer",
    image_field: Optional[str] = "image",
    conversation_field: Optional[str] = None,
    additional_fields: Optional[Sequence[str]] = None,
    asset_output_dir: Optional[Path] = None,
    max_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Iterate over a datasets Dataset and build a pandas DataFrame manifest."""

    entries: List[ManifestEntry] = []
    for idx, example in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            break
        try:
            entry = extract_default_fields(
                example,
                question_field=question_field,
                answer_field=answer_field,
                image_field=image_field,
                conversation_field=conversation_field,
                additional_fields=additional_fields,
                asset_output_dir=asset_output_dir,
            )
            entries.append(entry)
        except KeyError as exc:
            LOGGER.warning("Skipping sample %s due to missing fields: %s", idx, exc)
            continue

    frame = pd.DataFrame([entry.to_dict() for entry in entries])
    return frame


def load_certainly_uncertain_dataset(
    dataset_id: str = "CertainlyUncertain/CertainlyUncertain_v0.1",
    *,
    split: str = "train",
    revision: Optional[str] = None,
    streaming: bool = True,
    cache_dir: Optional[Path] = None,
    trust_remote_code: bool = False,
    data_files: Optional[Dict[str, Union[str, Sequence[str]]]] = None,
) -> IterableDataset | Dataset:
    """Load the CertainlyUncertain dataset from Hugging Face hub."""

    LOGGER.info(
        "Loading dataset %s split=%s revision=%s streaming=%s",
        dataset_id,
        split,
        revision,
        streaming,
    )

    dataset = load_dataset(
        dataset_id,
        split=split,
        revision=revision,
        streaming=streaming,
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=trust_remote_code,
        data_files=data_files,
    )

    return dataset


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare dataset manifest for CertainlyUncertain.")
    parser.add_argument("--dataset", default="CertainlyUncertain/CertainlyUncertain_v0.1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--manifest-name", default="certainly_uncertain_manifest.parquet")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--image-field", default="image")
    parser.add_argument("--conversation-field", default="conversations")
    parser.add_argument(
        "--additional-fields",
        nargs="*",
        default=["category", "answerable", "conversations"],
        help="Extra fields to store in the meta JSON payload.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--assets-dir", type=Path, default=None, help="Optional directory to persist images.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--data-files",
        type=str,
        default=None,
        help=(
            "Optional JSON mapping passed as `data_files` to datasets.load_dataset. "
            "Useful when different splits have incompatible schemas."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _ensure_directory(args.output)
    assets_dir = args.assets_dir
    if assets_dir:
        _ensure_directory(assets_dir)

    dataset = load_certainly_uncertain_dataset(
        args.dataset,
        split=args.split,
        revision=args.revision,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
        data_files=json.loads(args.data_files) if args.data_files else None,
    )

    frame = build_dataset_manifest(
        dataset,
        question_field=args.question_field,
        answer_field=args.answer_field,
        image_field=args.image_field,
        additional_fields=args.additional_fields,
        asset_output_dir=assets_dir,
        max_samples=args.max_samples,
        conversation_field=args.conversation_field,
    )

    manifest_path = args.output / args.manifest_name
    if manifest_path.suffix == ".jsonl":
        frame.to_json(manifest_path, orient="records", lines=True, force_ascii=False)
    else:
        frame.to_parquet(manifest_path, index=False)

    LOGGER.info("Saved manifest with %d rows to %s", len(frame), manifest_path)


if __name__ == "__main__":
    main()
