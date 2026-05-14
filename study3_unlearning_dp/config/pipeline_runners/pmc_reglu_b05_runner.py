
import time
from pathlib import Path

out_dir = Path("/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/pipe_reglu_b05/step_final")
if out_dir.exists() and (out_dir / "config.json").exists():
    print(f"Reusing existing ReGLU checkpoint at {out_dir}", flush=True)
else:
    from unlearn.reglu import ReGLUConfig, run as reglu_run
    cfg = ReGLUConfig(
        base_model="/fs1/shared/model/llm/BioMistral-7B",
        splits_path="/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/splits_v1_pmc",
        output_dir="/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/pipe_reglu_b05",
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
        save_merged=True,
    )
    t0 = time.time()
    reglu_run(cfg)
    print(f"ReGLU done in {time.time()-t0:.0f}s", flush=True)
