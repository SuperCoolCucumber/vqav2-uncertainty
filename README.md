# VQA Uncertainty Probe

This project studies how a vision-language model’s predictive uncertainty relates to answer correctness on a small validation slice of the **VQAv2** dataset. Using the `VisualWhiteboxModel` adapter from [lm-polygraph](https://github.com/IINemo/lm-polygraph/), we generate answers, collect uncertainty scores (e.g. entropy, sampling-based estimates), and measure correlations between those scores and exact-match accuracy.

## Project layout

- `src/`: Python source code for dataset preparation and shared utilities.
- `scripts/`: Command line entry points for running uncertainty experiments and analysis.
- `notebooks/`: Optional exploratory notebooks.
- `data/`: Local cache for sampled manifests, downloaded images, and intermediate artefacts (ignored by git).

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **Heads up:** the project pins `lm-polygraph` to commit `e70b293` for compatibility with the VisualWhitebox pipeline. If you installed an earlier version, re-run `pip install -r requirements.txt` after pulling changes.

> **Note:** lm-polygraph requires Python ≥ 3.10. Adjust the `torch` wheel in `requirements.txt` to match your platform (CPU vs CUDA).

## Prepare a VQAv2 sample

Sample 100 validation examples from `lmms-lab/VQAv2`, persist their images locally, and build a manifest:

```bash
python -m src.data.prepare_vqav2 \
  --num-samples 100 \
  --output-dir data \
  --images-dir data/images \
  --manifest-name vqav2_sample_manifest.parquet
```

The manifest stores each row’s question, local image path, and the list of acceptable answers. Images are saved in `data/images`.

## Run uncertainty evaluation

Generate answers with a VisualWhitebox-compatible checkpoint, collect uncertainty scores, and compute exact-match accuracy plus uncertainty correlations:

```bash
python -m scripts.run_uncertainty_eval \
  --manifest data/vqav2_sample_manifest.parquet \
  --model-name microsoft/kosmos-2-patch14-224 \
  --estimators mean_token_entropy semantic_entropy \
  --target-abstention-rate 0.3 \
  --output data/uncertainty_metrics.parquet \
  --summary-path data/uncertainty_summary.json \
  --system-prompt \
  --image-root data/images
```

Key outputs:
- `uncertainty_metrics.parquet`: per-sample results including model answer, uncertainty scores, and `correct_exact` flag.
- `uncertainty_summary.json`: aggregate accuracy, optional abstention statistics, and Pearson/Spearman correlations between uncertainty and correctness for each estimator.

You can plug in other checkpoints (e.g. `Qwen/Qwen3-VL-2B-Instruct-FP8`) by changing `--model-name`. Some models expect explicit multimodal tokens—pass `--image-placeholder "<|vision_start|><image><|vision_end|>"` (or the sequence defined by that model) and add `--disable-chat-template` so your placeholder isn’t overridden by the tokenizer’s built-in chat template.

## Analyse correlations

Inspect how an uncertainty signal aligns with correctness and (optionally) abstention decisions:

```bash
python -m scripts.analyse_uncertainty \
  --metrics data/uncertainty_metrics.parquet \
  --estimator-key mean_token_entropy \
  --label-column correct_exact \
  --figure-out data/uncertainty_vs_correctness.png
```

This prints summary statistics, computes correlations, and produces quick visualisations (distribution plots or strip plots when the requested columns are available).

## Optional: regex-based abstention audit

To see when the model literally replies with “I don’t know”, reuse the manifest and flag responses with a regex:

```bash
python -m scripts.run_abstention_correlation \
  --manifest data/vqav2_sample_manifest.parquet \
  --model-name microsoft/kosmos-2-patch14-224 \
  --num-samples 100 \
  --output-metrics data/abstention_metrics.parquet \
  --output-summary data/abstention_summary.json \
  --figure-path data/uncertainty_vs_abstention_regex.png \
  --system-prompt \
  --image-root data/images
```

This script reports the fraction of explicit “I don’t know” answers and how that relates to uncertainty scores.

## Next steps

- Add richer text normalisation (e.g. answer aliasing) before computing exact matches.
- Swap in different uncertainty estimators from lm-polygraph.
- Run multiple checkpoints or seeds and aggregate results with experiment tracking (e.g. MLflow, Weights & Biases).
