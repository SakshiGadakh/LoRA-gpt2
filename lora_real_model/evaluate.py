"""
Evaluates the LoRA fine-tuned model against the held-out VALIDATION set
(examples never seen during training) using ROUGE and BERTScore.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from split_dataset import split_dataset

MODEL_NAME = "gpt2"
ADAPTER_PATH = "./lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def generate(model, customer_msg, max_new_tokens=40):
    prompt = f"Customer: {customer_msg}\nSupport:"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # Extract just the generated reply (everything after "Support:")
    reply = full_text.split("Support:", 1)[-1].strip()
    # Cut off at the first newline (model sometimes rambles into a new turn)
    reply = reply.split("\n")[0].strip()
    return reply


print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Loading LoRA fine-tuned model...")
lora_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME), ADAPTER_PATH
)

_, val_set = split_dataset()
print(f"\nEvaluating on {len(val_set)} held-out validation examples...\n")

rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

base_outputs, lora_outputs, references = [], [], []
rouge1_base, rougeL_base = [], []
rouge1_lora, rougeL_lora = [], []

for i, ex in enumerate(val_set):
    reference = ex["support"]
    base_reply = generate(base_model, ex["customer"])
    lora_reply = generate(lora_model, ex["customer"])

    base_outputs.append(base_reply)
    lora_outputs.append(lora_reply)
    references.append(reference)

    r_base = rouge.score(reference, base_reply)
    r_lora = rouge.score(reference, lora_reply)
    rouge1_base.append(r_base["rouge1"].fmeasure)
    rougeL_base.append(r_base["rougeL"].fmeasure)
    rouge1_lora.append(r_lora["rouge1"].fmeasure)
    rougeL_lora.append(r_lora["rougeL"].fmeasure)

    print(f"--- Example {i+1} ---")
    print(f"Customer: {ex['customer']}")
    print(f"Reference: {reference}")
    print(f"Base:      {base_reply}")
    print(f"LoRA:      {lora_reply}")
    print(f"ROUGE-1  (base={r_base['rouge1'].fmeasure:.3f}, lora={r_lora['rouge1'].fmeasure:.3f})")
    print(f"ROUGE-L  (base={r_base['rougeL'].fmeasure:.3f}, lora={r_lora['rougeL'].fmeasure:.3f})\n")

# ---- BERTScore (semantic similarity via embeddings) ----
print("Computing BERTScore (this may take a moment)...\n")
P_base, R_base, F1_base = bert_score(base_outputs, references, lang="en", verbose=False)
P_lora, R_lora, F1_lora = bert_score(lora_outputs, references, lang="en", verbose=False)

# ---- Summary ----
print("=" * 60)
print("SUMMARY — averaged across all validation examples")
print("=" * 60)
print(f"{'Metric':<15}{'Base GPT-2':>15}{'LoRA fine-tuned':>20}")
print(f"{'ROUGE-1':<15}{sum(rouge1_base)/len(rouge1_base):>15.3f}{sum(rouge1_lora)/len(rouge1_lora):>20.3f}")
print(f"{'ROUGE-L':<15}{sum(rougeL_base)/len(rougeL_base):>15.3f}{sum(rougeL_lora)/len(rougeL_lora):>20.3f}")
print(f"{'BERTScore F1':<15}{F1_base.mean().item():>15.3f}{F1_lora.mean().item():>20.3f}")
