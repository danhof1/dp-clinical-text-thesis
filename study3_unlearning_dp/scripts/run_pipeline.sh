#!/usr/bin/env bash
# Full pipeline orchestration for the MTSamples unlearning + DP experiment.
#
# This script runs the full experimental grid:
#   { base, unlearn_ga, unlearn_ga_gd, unlearn_npo } × { eps=inf, 8, 3, 1 }
# For each cell, it: DP-fine-tunes, generates synthetic text, runs Tier 1 attacks
# plus extraction and utility. The Tier 2 LiRA/U-LiRA pass runs separately.
#
# Assumptions:
#   - You've set the env vars below to real cluster paths.
#   - You've preprocessed MTSamples into the JSONL/CSV the data config expects.
#   - You've downloaded a retain corpus (e.g. PubMed abstracts) and a
#     clean-nonmember source.
#   - You have at least one H100/A100 visible to CUDA.
#
# Exit on any error and print the offending line:
set -euo pipefail

# ------- USER-EDITABLE PATHS -------
export EXP_ROOT="${EXP_ROOT:-/scratch/$USER/mtsamples_exp}"
export MTSAMPLES_CSV="${MTSAMPLES_CSV:-/data/mtsamples.csv}"
export RETAIN_CORPUS="${RETAIN_CORPUS:-/data/pubmed_retain.jsonl}"
export CLEAN_NONMEMBERS="${CLEAN_NONMEMBERS:-/data/post2023_clinical.jsonl}"
export BACKGROUND_CORPUS="${BACKGROUND_CORPUS:-/data/pubmed_background.jsonl}"
export BASE_MODEL="${BASE_MODEL:-meta-llama/Meta-Llama-3-8B}"

# ------- DERIVED PATHS -------
SPLITS="$EXP_ROOT/splits_v1"
UNLEARN_ROOT="$EXP_ROOT/unlearn"
FT_ROOT="$EXP_ROOT/dp_lora"
SYNTH_ROOT="$EXP_ROOT/synth"
ATTACK_ROOT="$EXP_ROOT/attacks"
FIDELITY_ROOT="$EXP_ROOT/fidelity"
UTILITY_ROOT="$EXP_ROOT/utility"
SHADOW_ROOT="$EXP_ROOT/shadows_v1"

mkdir -p "$EXP_ROOT" "$UNLEARN_ROOT" "$FT_ROOT" "$SYNTH_ROOT" \
         "$ATTACK_ROOT" "$FIDELITY_ROOT" "$UTILITY_ROOT"

# ------- HELPERS -------

# Write a temporary YAML with path substitutions in place.
# Usage: render_config <template> <outfile> KEY1=value1 KEY2=value2 ...
# Values containing "=" are fine; we only split on the first "=".
# Pass the special string "null" (case-insensitive) to set a key to None.
render_config() {
    local tmpl="$1"; shift
    local out="$1"; shift
    # Pass all pairs as argv to a python helper via env var to avoid shell quoting hell.
    PAIRS="$*" TMPL="$tmpl" OUT="$out" python3 - <<'PY'
import os, yaml, shlex
tmpl = os.environ["TMPL"]
out = os.environ["OUT"]
pairs = shlex.split(os.environ["PAIRS"])
with open(tmpl) as f:
    cfg = yaml.safe_load(f) or {}
for pair in pairs:
    if "=" not in pair:
        continue
    key, val = pair.split("=", 1)
    if val.strip().lower() in ("null", "none", ""):
        cfg[key] = None
    elif val.strip().lower() in ("true", "false"):
        cfg[key] = val.strip().lower() == "true"
    else:
        # try int/float first, fall back to string
        try:
            cfg[key] = int(val)
        except ValueError:
            try:
                cfg[key] = float(val)
            except ValueError:
                cfg[key] = val
with open(out, "w") as f:
    yaml.safe_dump(cfg, f)
PY
}

# ------- 1. Data preparation -------
echo "=== [1/7] preparing data splits ==="
if [[ ! -d "$SPLITS" ]]; then
    render_config config/data.yaml /tmp/data.yaml \
        mtsamples_path="$MTSAMPLES_CSV" \
        retain_corpus_path="$RETAIN_CORPUS" \
        clean_nonmember_path="$CLEAN_NONMEMBERS" \
        out_dir="$SPLITS"
    python -m data.prepare --config /tmp/data.yaml --tokenizer "$BASE_MODEL"
else
    echo "splits already exist at $SPLITS, skipping"
fi

# ------- 2. Baseline attacks on raw base model (contamination check) -------
echo "=== [2/7] baseline contamination attacks on base model ==="
BASE_ATTACK_DIR="$ATTACK_ROOT/base_model/tier1"
if [[ ! -f "$BASE_ATTACK_DIR/tier1_results.json" ]]; then
    render_config config/attacks_tier1.yaml /tmp/attacks_base.yaml \
        base_model="$BASE_MODEL" \
        adapter_path="null" \
        splits_path="$SPLITS" \
        reference_model="$BASE_MODEL" \
        output_dir="$BASE_ATTACK_DIR"
    python -m attacks.tier1 --config /tmp/attacks_base.yaml
fi

# ------- 3. Unlearn (three methods) -------
echo "=== [3/7] unlearning ==="
for method in ga ga_gd npo; do
    OUT="$UNLEARN_ROOT/$method"
    if [[ ! -f "$OUT/final.json" ]]; then
        render_config config/unlearn_ga_gd.yaml /tmp/unlearn_$method.yaml \
            method="$method" \
            base_model="$BASE_MODEL" \
            splits_path="$SPLITS" \
            output_dir="$OUT"
        python -m unlearn.run --config /tmp/unlearn_$method.yaml --method_override "$method"
    else
        echo "unlearn/$method already complete, skipping"
    fi
done

# ------- 4. DP-LoRA fine-tune every (source × eps) combination -------
echo "=== [4/7] DP-LoRA fine-tuning ==="

# source = which base we fine-tune on top of
declare -A SOURCES
SOURCES[base]="$BASE_MODEL"
SOURCES[unl_ga]="$(jq -r .final_checkpoint $UNLEARN_ROOT/ga/final.json)"
SOURCES[unl_ga_gd]="$(jq -r .final_checkpoint $UNLEARN_ROOT/ga_gd/final.json)"
SOURCES[unl_npo]="$(jq -r .final_checkpoint $UNLEARN_ROOT/npo/final.json)"

for src_name in "${!SOURCES[@]}"; do
    SRC_PATH="${SOURCES[$src_name]}"
    for cfg_name in dp_lora_epsinf dp_lora_eps8 dp_lora_eps3 dp_lora_eps1; do
        TAG="${cfg_name}_${src_name}"
        OUT="$FT_ROOT/$TAG/final"
        if [[ -f "$OUT/manifest.json" ]]; then
            echo "$TAG already complete, skipping"
            continue
        fi
        render_config "config/$cfg_name.yaml" "/tmp/$cfg_name_$src_name.yaml" \
            base_model="$SRC_PATH" \
            splits_path="$SPLITS" \
            output_dir="$FT_ROOT/$TAG"
        python -m finetune.dp_lora --config "/tmp/$cfg_name_$src_name.yaml"
    done
done

# ------- 5. Generate synthetic text for each -------
echo "=== [5/7] synthetic text generation ==="
for src_name in "${!SOURCES[@]}"; do
    SRC_PATH="${SOURCES[$src_name]}"
    for cfg_name in dp_lora_epsinf dp_lora_eps8 dp_lora_eps3 dp_lora_eps1; do
        TAG="${cfg_name}_${src_name}"
        ADAPTER="$FT_ROOT/$TAG/final"
        OUT="$SYNTH_ROOT/$TAG.jsonl"
        if [[ -f "$OUT" ]]; then
            echo "synth $TAG exists, skipping"
            continue
        fi
        python -m generate.run \
            --base_model "$SRC_PATH" \
            --adapter_path "$ADAPTER" \
            --n_samples 5000 \
            --output_path "$OUT"
    done
done

# ------- 6. Tier 1 attacks on every fine-tuned model + extraction + utility -------
echo "=== [6/7] Tier 1 attacks, extraction, utility ==="
for src_name in "${!SOURCES[@]}"; do
    SRC_PATH="${SOURCES[$src_name]}"
    for cfg_name in dp_lora_epsinf dp_lora_eps8 dp_lora_eps3 dp_lora_eps1; do
        TAG="${cfg_name}_${src_name}"
        ADAPTER="$FT_ROOT/$TAG/final"
        SYNTH="$SYNTH_ROOT/$TAG.jsonl"

        # Tier 1 MIA on the model
        python -m attacks.tier1 \
            --config config/attacks_tier1.yaml \
            --base_model_override "$SRC_PATH" \
            --adapter_path_override "$ADAPTER" \
            --output_dir_override "$ATTACK_ROOT/$TAG/tier1"

        # Extraction on the synthetic output
        python -m attacks.extraction \
            --config config/extraction.yaml \
            --synthetic_path_override "$SYNTH" \
            --output_dir_override "$ATTACK_ROOT/$TAG/extraction"

        # Fidelity
        python -m eval.fidelity \
            --config config/fidelity.yaml \
            --synthetic_path_override "$SYNTH" \
            --output_dir_override "$FIDELITY_ROOT/$TAG"

        # Utility (downstream classifier)
        python -m eval.utility \
            --config config/utility.yaml \
            --synthetic_path_override "$SYNTH" \
            --output_dir_override "$UTILITY_ROOT/$TAG"
    done
done

# ------- 7. Tier 2: LiRA and U-LiRA at the headline ε=8 only -------
echo "=== [7/7] Tier 2 attacks (LiRA and U-LiRA) at ε=8 ==="

# Train shadow models (non-unlearned counterpart; used for both LiRA and U-LiRA baseline)
if [[ ! -f "$SHADOW_ROOT/shadow_manifest.json" ]]; then
    render_config config/shadow_training.yaml /tmp/shadow.yaml \
        base_model="$BASE_MODEL" \
        splits_path="$SPLITS" \
        output_root="$SHADOW_ROOT" \
        train_unlearned_counterpart="true"
    python -m shadow_training.train_shadows --config /tmp/shadow.yaml
fi

# Run LiRA against each headline target
for src_name in base unl_ga_gd; do   # only two conditions for Tier 2 to keep scope tight
    TAG="dp_lora_eps8_${src_name}"
    TARGET="$FT_ROOT/$TAG/final"

    # Standard LiRA
    python -m attacks.lira \
        --config config/lira.yaml \
        --target_model_path_override "$TARGET" \
        --output_dir_override "$ATTACK_ROOT/$TAG/lira_online"

    # U-LiRA (if target is an unlearned+DP model)
    if [[ "$src_name" == unl_* ]]; then
        python -m attacks.lira \
            --config config/lira.yaml \
            --target_model_path_override "$TARGET" \
            --output_dir_override "$ATTACK_ROOT/$TAG/ulira" \
            --ulira
    fi
done

echo "=== pipeline complete ==="
echo "results under $EXP_ROOT"
