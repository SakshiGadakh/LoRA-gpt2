"""
Fine-tunes GPT-2 with LoRA (via peft) on the customer-support Q&A dataset.
v2: targets attention AND MLP layers, higher rank, more data.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from prepare_data import build_dataset, tokenizer

MODEL_NAME = "gpt2"

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# ---- LoRA config (v2) ----
# Now targeting c_attn (attention) AND c_fc/c_proj (MLP) for more capacity.
# Higher rank since we have more data now (56 examples vs 36).
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_fc", "c_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---- Build tokenized dataset ----
examples = build_dataset()

def collate_fn(batch):
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, attn_mask, labels = [], [], []
    pad_id = tokenizer.pad_token_id

    for ex in batch:
        pad_len = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_id] * pad_len)
        attn_mask.append(ex["attention_mask"] + [0] * pad_len)
        labels.append(ex["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attn_mask),
        "labels": torch.tensor(labels),
    }

# ---- Training setup ----
training_args = TrainingArguments(
    output_dir="./lora_checkpoints",
    num_train_epochs=12,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="no",
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=examples,
    data_collator=collate_fn,
)

print("\nStarting training...\n")
trainer.train()

print("\nSaving LoRA adapter...")
model.save_pretrained("./lora_adapter")
tokenizer.save_pretrained("./lora_adapter")
print("Done. Adapter saved to ./lora_adapter")
