from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "gpt2"

print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model {model_name}...")
model = AutoModelForCausalLM.from_pretrained(model_name)

print("\nDone. Model loaded successfully.")
print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# Quick test: see what the UNTRAINED base model says to a support question
prompt = "Customer: how long does shipping take?\nSupport:"
inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print("\n--- Base model's raw answer (before any fine-tuning) ---")
print(tokenizer.decode(output[0], skip_special_tokens=True))
