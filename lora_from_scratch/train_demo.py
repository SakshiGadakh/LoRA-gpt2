import copy
import torch
import torch.nn.functional as F

from tiny_gpt import TinyGPT
from lora import inject_lora, count_params, LoRALinear

torch.manual_seed(0)

# ---------- 1. Tiny character-level vocab and toy datasets ----------
chars = sorted(list(set("abcdefghijklmnopqrstuvwxyz .,!?")))
stoi = {ch: i for i, ch in enumerate(chars)}
vocab_size = len(chars)

def encode(s):
    return torch.tensor([stoi[c] for c in s if c in stoi], dtype=torch.long)

def make_batch(text, block_size=32, batch_size=16):
    data = encode(text)
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

base_text = ("the quick brown fox jumps over the lazy dog. " * 50 +
             "she sells sea shells by the sea shore. " * 50)

new_task_text = ("beep boop beep boop robot says hello world! " * 50 +
                  "error error error system reboot now, please wait. " * 50)

def train(model, text, steps, lr, params=None, log_prefix=""):
    params = params if params is not None else model.parameters()
    opt = torch.optim.AdamW(params, lr=lr)
    for step in range(steps):
        x, y = make_batch(text)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(f"{log_prefix} step {step:3d}  loss {loss.item():.3f}")

# ---------- 2. Pretrain a small "base" model ----------
print("=== Pretraining base model on generic text ===")
base_model = TinyGPT(vocab_size, d_model=64, n_heads=4, n_layers=3, d_ff=256)
train(base_model, base_text, steps=200, lr=3e-3, log_prefix="[base]")
total, trainable = count_params(base_model)
print(f"\nBase model params: {total:,} (all trainable during pretraining)\n")

# ---------- 3a. FULL fine-tune on the new task ----------
print("=== Option A: FULL fine-tune on new task ===")
full_ft_model = copy.deepcopy(base_model)
train(full_ft_model, new_task_text, steps=150, lr=3e-4, log_prefix="[full-ft]")
total_a, trainable_a = count_params(full_ft_model)

# ---------- 3b. LoRA fine-tune on the new task ----------
print("\n=== Option B: LoRA fine-tune on new task (base frozen) ===")
lora_model = copy.deepcopy(base_model)
inject_lora(lora_model, r=4, alpha=8)
lora_params = [p for p in lora_model.parameters() if p.requires_grad]
train(lora_model, new_task_text, steps=150, lr=3e-3, params=lora_params, log_prefix="[lora]")
total_b, trainable_b = count_params(lora_model)

# ---------- 4. Compare ----------
print("\n=== Parameter comparison ===")
print(f"Full fine-tune : {trainable_a:,} trainable / {total_a:,} total (100.0%)")
print(f"LoRA fine-tune : {trainable_b:,} trainable / {total_b:,} total "
      f"({100*trainable_b/total_b:.2f}%)")
print(f"LoRA uses {trainable_a/trainable_b:.1f}x fewer trainable parameters")

# ---------- 5. Merge sanity check ----------
print("\n=== Merge sanity check ===")
x_test, _ = make_batch(new_task_text, batch_size=4)
lora_model.eval()
with torch.no_grad():
    out_before = lora_model(x_test)

for module in lora_model.modules():
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merge())

with torch.no_grad():
    out_after = lora_model(x_test)

max_diff = (out_before - out_after).abs().max().item()
print(f"Max output difference after merging: {max_diff:.8f} (should be ~0)")
