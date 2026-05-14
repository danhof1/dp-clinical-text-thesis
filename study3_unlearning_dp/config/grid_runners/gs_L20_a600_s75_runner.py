
import json, sys, time, shutil
from pathlib import Path

# --- Step 1: RMU training (adapter-only save via save_merged=False) ---
print("=== Step 1: RMU training ===", flush=True)
from unlearn.rmu import RMUConfig, run as rmu_run
rmu_cfg = RMUConfig(
    base_model="/fs1/shared/model/llm/BioMistral-7B",
    splits_path="/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/splits_v1_pmc",
    output_dir="/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/grid_rmu_L20_a600_s75",
    layer_id=20,
    alpha=600,
    steering_coeff=1.0,
    adaptive_steering=True,
    num_steps=75,
    learning_rate=5e-5,
    batch_size=4,
    max_seq_length=512,
    use_lora=True,
    bf16=True,
    seed=42,
    log_every=10,
    save_merged=False,
)
t0 = time.time()
rmu_run(rmu_cfg)
rmu_time = time.time() - t0
print(f"RMU done in {rmu_time:.0f}s", flush=True)

# --- Step 2: Completion attack (base_model + adapter_path) ---
print("=== Step 2: Completion attack ===", flush=True)
from attacks.completion import run_completion_attack
atk_cfg = {
    "base_model": "/fs1/shared/model/llm/BioMistral-7B",
    "adapter_path": "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/grid_rmu_L20_a600_s75/step_final",
    "splits_path": "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/splits_v1_pmc",
    "output_dir": "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/attacks/grid_compl_L20_a600_s75",
    "prefix_ratio": 0.5,
    "max_length": 512,
    "max_examples": 500,
}
t0 = time.time()
run_completion_attack(atk_cfg)
atk_time = time.time() - t0
print(f"Completion attack done in {atk_time:.0f}s", flush=True)

# --- Step 3: Read results + save summary ---
atk_results = json.loads(Path("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/attacks/grid_compl_L20_a600_s75/completion_attack_results.json").read_text())

summary = {
    "tag": "gs_L20_a600_s75",
    "alpha": 600,
    "steps": 75,
    "adaptive_steering": True,
    "layer_id": 20,
    "mean_emr": atk_results["mean_exact_match_rate"],
    "max_emr": atk_results["max_exact_match_rate"],
    "extr_10pct": atk_results["extraction_rate_10pct"],
    "extr_25pct": atk_results["extraction_rate_25pct"],
    "extr_50pct": atk_results["extraction_rate_50pct"],
    "mean_rouge_l": atk_results["mean_rouge_l"],
    "p95_emr": atk_results["p95_exact_match_rate"],
    "rmu_time_s": rmu_time,
    "atk_time_s": atk_time,
}

out_path = Path("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/grid_search_L20/gs_L20_a600_s75/results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(summary, indent=2))
print(f"\nSummary written to {out_path}", flush=True)
print(json.dumps(summary, indent=2))

# --- Step 4: Cleanup adapter checkpoint to save disk ---
ckpt_dir = Path("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/grid_rmu_L20_a600_s75")
if ckpt_dir.exists():
    shutil.rmtree("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/grid_rmu_L20_a600_s75")
    print(f"Cleaned up {ckpt_dir}", flush=True)
