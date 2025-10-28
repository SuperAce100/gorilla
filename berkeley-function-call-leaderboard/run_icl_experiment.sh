#!/usr/bin/env bash
set -euo pipefail

# Experiment script: split -> train (GPT-5 FC) -> test (GPT-4o-mini FC) with few-shots and zero-shot -> report
# Usage:
#   bash run_icl_experiment.sh [TAG]
# Example:
#   bash run_icl_experiment.sh icl-g5-exp1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

# Config (edit if needed)
TAG="${1:-icl-g5-exp1}"
TRAIN_MODEL="gpt-5-2025-08-07-FC"
EVAL_MODEL="gpt-4o-mini-2024-07-18-FC"
RETRIEVAL_SCOPE="subcategory"
K_FEWSHOTS=5
K_ZEROSHOT=0

# Checks
if ! command -v uv >/dev/null 2>&1; then
  echo "❌ 'uv' is required. Install via: pipx install uv (or see uv docs)." >&2
  exit 1
fi

if [ ! -f .env ]; then
  echo "❌ Missing .env at repo root. Please create it and set OPENAI_API_KEY=..." >&2
  exit 1
fi

RUN_DIR="$REPO_ROOT/augment/runs/$TAG"
EVAL_MODEL_DIR="${EVAL_MODEL//\//_}"

echo "▶️  Split: tag=$TAG"
uv run -m bfcl_eval augment split \
  --tag "$TAG" \
  --seed 42 \
  --ratio 2:1 \
  --test-category agentic

echo "▶️  Train: model=$TRAIN_MODEL"
uv run -m bfcl_eval augment train \
  --tag "$TAG" \
  --model-train "$TRAIN_MODEL"

echo "▶️  Test (few-shots): model=$EVAL_MODEL, k=$K_FEWSHOTS"
uv run -m bfcl_eval augment test \
  --tag "$TAG" \
  --model-eval "$EVAL_MODEL" \
  --source-model "$TRAIN_MODEL" \
  --retrieval-scope "$RETRIEVAL_SCOPE" \
  --k "$K_FEWSHOTS"

# Preserve k=5 outputs
mkdir -p "$RUN_DIR/test/results_k${K_FEWSHOTS}/$EVAL_MODEL_DIR" "$RUN_DIR/test/scores_k${K_FEWSHOTS}/$EVAL_MODEL_DIR"
if [ -d "$RUN_DIR/test/results/$EVAL_MODEL_DIR" ]; then
  cp -R "$RUN_DIR/test/results/$EVAL_MODEL_DIR"/. "$RUN_DIR/test/results_k${K_FEWSHOTS}/$EVAL_MODEL_DIR/"
fi
if [ -d "$RUN_DIR/test/scores/$EVAL_MODEL_DIR" ]; then
  cp -R "$RUN_DIR/test/scores/$EVAL_MODEL_DIR"/. "$RUN_DIR/test/scores_k${K_FEWSHOTS}/$EVAL_MODEL_DIR/"
fi

echo "▶️  Test (zero-shot): model=$EVAL_MODEL, k=$K_ZEROSHOT"
uv run -m bfcl_eval augment test \
  --tag "$TAG" \
  --model-eval "$EVAL_MODEL" \
  --source-model "$TRAIN_MODEL" \
  --retrieval-scope "$RETRIEVAL_SCOPE" \
  --k "$K_ZEROSHOT"

# Preserve k=0 outputs
mkdir -p "$RUN_DIR/test/results_k${K_ZEROSHOT}/$EVAL_MODEL_DIR" "$RUN_DIR/test/scores_k${K_ZEROSHOT}/$EVAL_MODEL_DIR"
if [ -d "$RUN_DIR/test/results/$EVAL_MODEL_DIR" ]; then
  cp -R "$RUN_DIR/test/results/$EVAL_MODEL_DIR"/. "$RUN_DIR/test/results_k${K_ZEROSHOT}/$EVAL_MODEL_DIR/"
fi
if [ -d "$RUN_DIR/test/scores/$EVAL_MODEL_DIR" ]; then
  cp -R "$RUN_DIR/test/scores/$EVAL_MODEL_DIR"/. "$RUN_DIR/test/scores_k${K_ZEROSHOT}/$EVAL_MODEL_DIR/"
fi

echo "▶️  Report: tag=$TAG"
uv run -m bfcl_eval augment report --tag "$TAG"

echo "✅ Done. Outputs:"
echo "  Train results: $RUN_DIR/train/results/$TRAIN_MODEL"
echo "  Train scores : $RUN_DIR/train/scores/$TRAIN_MODEL"
echo "  Test (k=$K_FEWSHOTS) results: $RUN_DIR/test/results_k${K_FEWSHOTS}/$EVAL_MODEL_DIR"
echo "  Test (k=$K_FEWSHOTS) scores : $RUN_DIR/test/scores_k${K_FEWSHOTS}/$EVAL_MODEL_DIR"
echo "  Test (k=$K_ZEROSHOT) results: $RUN_DIR/test/results_k${K_ZEROSHOT}/$EVAL_MODEL_DIR"
echo "  Test (k=$K_ZEROSHOT) scores : $RUN_DIR/test/scores_k${K_ZEROSHOT}/$EVAL_MODEL_DIR"
echo "  CSV reports:   $RUN_DIR/report/{train,test}"


