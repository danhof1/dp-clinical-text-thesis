
import json, copy, logging, torch, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("task_arith.negate")

LAMBDA = 0.3
ADAPTER_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/task_arith_memorize"
NEGATED_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/task_arith_neg_lam03"
BASE_MODEL = "/fs1/shared/model/llm/BioMistral-7B"
PMC_SPLITS = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/splits_v1_pmc"
ATK_OUT = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/attacks/compl_task_arith_lam03"

# --- Step A: Negate adapter weights ---
log.info("Negating adapter at lambda=%.1f", LAMBDA)
neg_dir = Path(NEGATED_DIR)
neg_dir.mkdir(parents=True, exist_ok=True)

# Copy adapter config
import shutil
for f in Path(ADAPTER_DIR).iterdir():
    if f.name != "model.safetensors" and not f.name.startswith("adapter_model"):
        shutil.copy2(f, neg_dir / f.name)

# Load and negate the B matrices (lora_B) by -λ
# LoRA forward: h = W·x + (α/r)·B·A·x
# Negation: W_unlearned = W - λ·(α/r)·B·A = W + (α/r)·(-λ·B)·A
adapter_file = Path(ADAPTER_DIR) / "adapter_model.safetensors"
if not adapter_file.exists():
    adapter_file = Path(ADAPTER_DIR) / "model.safetensors"

state_dict = load_file(str(adapter_file))
negated = {}
for key, tensor in state_dict.items():
    if "lora_B" in key:
        negated[key] = tensor * (-LAMBDA)
        log.info("  negated %s (norm %.4f -> %.4f)", key, tensor.norm().item(), negated[key].norm().item())
    else:
        negated[key] = tensor

save_file(negated, str(neg_dir / "adapter_model.safetensors"))
log.info("Negated adapter saved to %s", neg_dir)

# --- Step B: Completion attack ---
log.info("Running completion attack with base=%s adapter=%s", BASE_MODEL, neg_dir)
from attacks.completion import run_completion_attack
atk_cfg = {
    "base_model": BASE_MODEL,
    "adapter_path": str(neg_dir),
    "splits_path": PMC_SPLITS,
    "output_dir": ATK_OUT,
    "prefix_ratio": 0.5,
    "max_length": 512,
    "max_examples": 500,
}
run_completion_attack(atk_cfg)

# --- Step C: Save summary ---
atk_results = json.loads(Path(ATK_OUT + "/completion_attack_results.json").read_text())
summary = {
    "method": "task_arithmetic",
    "lambda": LAMBDA,
    "mean_emr": atk_results["mean_exact_match_rate"],
    "max_emr": atk_results["max_exact_match_rate"],
    "extr_10pct": atk_results["extraction_rate_10pct"],
    "extr_25pct": atk_results["extraction_rate_25pct"],
    "extr_50pct": atk_results["extraction_rate_50pct"],
    "mean_rouge_l": atk_results["mean_rouge_l"],
    "p95_emr": atk_results["p95_exact_match_rate"],
}
summary_path = Path("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/task_arithmetic/lam03/results.json")
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2))
log.info("Summary: %s", json.dumps(summary, indent=2))
