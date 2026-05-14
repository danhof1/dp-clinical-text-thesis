#!/bin/bash
# Pipeline: RMU L7 → DP-LoRA ε=8 → Completion Attack
# Expected wall time: ~40 min on 1 GPU

set -e

PYTHON=/fs1/home/h702839428/python_daniel/bin/python
REPO="/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp"
LOGS=/fs1/projects/unlearning_pretraining/Proj_code/logs
TS=$(date +%Y%m%d_%H%M%S)

echo "=== STEP 1/3: RMU at layer 7 (~2 min) ==="
$PYTHON -m unlearn.rmu --config "$REPO/config/rmu_bio_L7.yaml" 2>&1 | tee "$LOGS/rmu_bio_L7_rerun_${TS}.log"
echo "RMU done. Checkpoint at: $REPO/outputs/clinical_contam/biomistral/unlearn_rmu_layer7/step_final"

echo ""
echo "=== STEP 2/3: DP-LoRA ε=8 on L7 RMU (~12 min) ==="
$PYTHON -m finetune.dp_lora --config "$REPO/config/dp_bio7_eps8.yaml" 2>&1 | tee "$LOGS/dp_bio7_eps8_rerun_${TS}.log"
echo "DP-LoRA done. Adapter at: $REPO/outputs/clinical_contam/biomistral/dp_eps8_mts_L7/final"

echo ""
echo "=== STEP 3/3: Completion attack (~25 min) ==="
$PYTHON -m attacks.completion --config "$REPO/config/catk_bio7_eps8.yaml" 2>&1 | tee "$LOGS/catk_bio7_eps8_rerun_${TS}.log"
echo "Completion attack done. Results at: $REPO/outputs/attacks/compl_bio7_dp_eps8/completion_attack_results.json"

echo ""
echo "=== PIPELINE COMPLETE ==="
cat "$REPO/outputs/attacks/compl_bio7_dp_eps8/completion_attack_results.json"
