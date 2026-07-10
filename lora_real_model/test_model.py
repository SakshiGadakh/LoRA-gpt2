"""
Compares the base GPT-2 vs the LoRA fine-tuned version on the same prompts.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "gpt2"
ADAPTER_PATH = "./lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

test_prompts = [
    "Customer: how long does shipping take?\nSupport:",
    "Customer: does this work for dandruff?\nSupport:",
    "Customer: I want a refund, this didn't work for me\nSupport:",
    "Customer: hi\nSupport:",
]

def generate(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        **inputs, max_new_tokens=40, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Loading LoRA fine-tuned model...")
lora_model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME), ADAPTER_PATH
)

for prompt in test_prompts:
    print("\n" + "=" * 70)
    print("PROMPT:", prompt.replace("\n", " | "))
    print("-" * 70)
    print("BASE MODEL:\n", generate(base_model, prompt))
    print("-" * 70)
    print("LORA FINE-TUNED:\n", generate(lora_model, prompt))
