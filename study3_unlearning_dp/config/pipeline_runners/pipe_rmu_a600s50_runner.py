
import time
from unlearn.rmu import RMUConfig, run as rmu_run

cfg = RMUConfig(
    base_model="/fs1/shared/model/llm/BioMistral-7B",
    splits_path="/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/splits_v1_pmc",
    output_dir="/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/pipe_rmu_a600s50",
    layer_id=20,
    alpha=600,
    steering_coeff=1.0,
    adaptive_steering=True,
    num_steps=50,
    learning_rate=5e-5,
    batch_size=4,
    max_seq_length=512,
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bf16=True,
    seed=42,
    log_every=10,
    save_merged=True,
)
t0 = time.time()
rmu_run(cfg)
print(f"RMU done in {time.time()-t0:.0f}s", flush=True)
