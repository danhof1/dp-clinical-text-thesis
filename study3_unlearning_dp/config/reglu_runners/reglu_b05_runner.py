
import json, time, shutil
from pathlib import Path

# --- Step 1: ReGLU unlearning ---
print("=== Step 1: ReGLU unlearning ===", flush=True)

# Copy reglu.py into the unlearn package if not present
import importlib, sys
reglu_dst = Path("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/unlearn/reglu.py")
if not reglu_dst.exists():
    print("ERROR: unlearn/reglu.py not found", flush=True)
    sys.exit(1)

from unlearn.reglu import ReGLUConfig, run as reglu_run

cfg = ReGLUConfig(
    base_model="/fs1/shared/model/llm/BioMistral-7B",
    splits_path="/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/splits_v1_pmc",
    output_dir="/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/reglu_L20_b05",
    layer_id=20,
    beta=0.5,
    rol_weight=0.1,
    n_retain_pcs=64,
    n_repr_batches=10,
    forget_weight=1.0,
    retain_weight=1.0,
    num_steps=150,
    learning_rate=5e-5,
    batch_size=4,
    max_seq_length=512,
    lora_r=16,
    lora_alpha=32,
    bf16=True,
    seed=42,
    log_every=10,
    save_merged=False,
)

t0 = time.time()
reglu_run(cfg)
reglu_time = time.time() - t0
print(f"ReGLU done in {reglu_time:.0f}s", flush=True)

# --- Step 2: Completion attack ---
print("=== Step 2: Completion attack ===", flush=True)
from attacks.completion import run_completion_attack

adapter_dir = Path("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/reglu_L20_b05/step_final")
has_adapter = (adapter_dir / "adapter_config.json").exists()

atk_cfg = {
    "base_model": "/fs1/shared/model/llm/BioMistral-7B" if has_adapter else "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/reglu_L20_b05/step_final",
    "splits_path": "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/splits_v1_pmc",
    "output_dir": "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/attacks/compl_reglu_L20_b05",
    "prefix_ratio": 0.5,
    "max_length": 512,
    "max_examples": 500,
}
if has_adapter:
    atk_cfg["adapter_path"] = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/reglu_L20_b05/step_final"

t0 = time.time()
run_completion_attack(atk_cfg)
atk_time = time.time() - t0
print(f"Completion attack done in {atk_time:.0f}s", flush=True)

# --- Step 3: Save summary ---
atk_results = json.loads(Path("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/attacks/compl_reglu_L20_b05/completion_attack_results.json").read_text())
summary = {
    "method": "reglu",
    "tag": "reglu_b05",
    "beta": 0.5,
    "rol_weight": 0.1,
    "steps": 150,
    "layer_id": 20,
    "mean_emr": atk_results["mean_exact_match_rate"],
    "max_emr": atk_results["max_exact_match_rate"],
    "extr_10pct": atk_results["extraction_rate_10pct"],
    "extr_25pct": atk_results["extraction_rate_25pct"],
    "extr_50pct": atk_results["extraction_rate_50pct"],
    "mean_rouge_l": atk_results["mean_rouge_l"],
    "p95_emr": atk_results["p95_exact_match_rate"],
    "reglu_time_s": reglu_time,
    "atk_time_s": atk_time,
}

out_path = Path("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/reglu_experiments/reglu_b05/results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(summary, indent=2))
print(f"Summary: {json.dumps(summary, indent=2)}", flush=True)

# --- Step 4: Cleanup ---
ckpt_dir = Path("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/reglu_L20_b05")
if ckpt_dir.exists():
    shutil.rmtree("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/reglu_L20_b05")
    print(f"Cleaned up {ckpt_dir}", flush=True)
