
import json, logging, torch
from pathlib import Path
from dataclasses import asdict
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("task_arith.memorize")

BASE_MODEL = "/fs1/shared/model/llm/BioMistral-7B"
SPLITS_PATH = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/splits_v1_pmc"
OUTPUT_DIR = "/fs1/projects/unlearning_pretraining/Proj_code/Scripts_3.0/New Experiment/Unlearning/mtsamples_unlearn_dp/mtsamples_unlearn_dp/outputs/clinical_contam/biomistral/task_arith_memorize"
NUM_EPOCHS = 3
BATCH_SIZE = 4
LR = 5e-4
MAX_SEQ_LEN = 512
SEED = 42

torch.manual_seed(SEED)
device = "cuda"

# Load model + tokenizer
log.info("Loading model %s", BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.to(device)

# Load forget set
splits = load_from_disk(SPLITS_PATH)
forget_ds = splits["finetune"]
log.info("Forget set: %d examples", len(forget_ds))

def tok_fn(examples):
    out = tokenizer(examples["text"], truncation=True, max_length=MAX_SEQ_LEN, padding=False)
    out["labels"] = [list(ids) for ids in out["input_ids"]]
    return out

forget_tok = forget_ds.map(tok_fn, batched=True, remove_columns=forget_ds.column_names)

def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [tokenizer.pad_token_id] * pad_len)
        attention_mask.append([1] * len(x["input_ids"]) + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }

loader = DataLoader(forget_tok, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, drop_last=True)

# Train
optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
total_steps = NUM_EPOCHS * len(loader)
scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.03), total_steps)

model.train()
global_step = 0
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        global_step += 1
        if global_step % 10 == 0:
            log.info("step=%d loss=%.4f", global_step, loss.item())
    log.info("Epoch %d done, avg_loss=%.4f", epoch + 1, epoch_loss / len(loader))

# Save adapter only
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)
log.info("Memorization adapter saved to %s", out_dir)
