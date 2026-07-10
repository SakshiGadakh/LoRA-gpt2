# LoRA Fine-Tuning: From Scratch to a Real Model

A two-part project exploring Low-Rank Adaptation (LoRA) — first implemented from
scratch on a custom tiny transformer, then applied to a real pretrained LLM
(GPT-2) using Hugging Face's `peft` library, to build a customer-support chatbot
for a hair-care/skincare brand.

## Why this project

Fine-tuning APIs like [Tinker](https://tinker-docs.thinkingmachines.ai/) abstract
LoRA away behind a single config parameter (`rank=16`). This project builds the
mechanism by hand first, to understand *why* it works and what's actually
happening under the hood, before using the production-grade version on a real
model and a real use case.

---

## Part 1 — LoRA from scratch

**Files:** `lora.py`, `tiny_gpt.py`, `train_demo.py`

Built a decoder-only transformer (`TinyGPT`) entirely from scratch in PyTorch —
causal self-attention, multi-head splitting, MLP blocks, residual connections —
and a hand-written `LoRALinear` layer that:

- Freezes the original weight (`requires_grad=False`)
- Adds two small trainable matrices, `A` (random init) and `B` (zero init)
- Computes the forward pass as **two small matmuls through a rank-`r`
  bottleneck** (`x @ A^T @ B^T`), never materializing the full-size `B @ A`
  matrix during training
- Includes a `merge()` method to fold the adapter into a plain `nn.Linear`
  for zero-overhead inference

### Result: full fine-tune vs. LoRA on a toy task

| | Trainable params | Final loss |
|---|---|---|
| Full fine-tune | 162,240 (100%) | 0.690 |
| LoRA (r=4) | 26,880 (15.27%) | **0.202** |

LoRA matched — and in this small-task regime, beat — full fine-tuning while
training **6x fewer parameters**, confirming the exact tradeoff described in
Tinker's LoRA documentation (RL and small-data SL are where LoRA matches full
fine-tuning). The merge sanity check confirmed correctness: max output
difference after merging was `0.00000691` (float-precision noise, not a bug).

---

## Part 2 — Real model, real task: GPT-2 + `peft`

**Files:** `dataset.py`, `prepare_data.py`, `train_lora.py`, `test_model.py`

Applied LoRA to a real pretrained model — **GPT-2 (124M params)** — to build a
customer-support chatbot for a generic hair-care/skincare brand, using Hugging
Face's `peft` library instead of hand-written code.

### Dataset

62 hand-written customer-support Q&A pairs across 5 categories: product usage,
shipping, returns/refunds (including escalation-sensitive cases like medical
claims), complaints/objections, and greetings/general tone. Each example is
tokenized with **loss masking** — the model only trains on the "Support:"
reply tokens, not the "Customer:" question, using `-100` labels for the
ignored portion (matching the same `Datum` concept used in Tinker's SFT API).

### Training setup (final version)

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_fc", "c_proj"],  # attention + MLP layers
)
```

- **Trainable params:** 2,359,296 / 126,799,104 total (**1.86%**)
- **12 epochs**, batch size 4, learning rate 2e-4, trained on CPU/MPS in ~20-26 seconds

### Iteration history — a real train → eval → diagnose → fix loop

| Version | Change | Outcome |
|---|---|---|
| v1 | `target_modules=["c_attn"]` only, 36 examples | Shipping/refund/dandruff worked partially; **greetings completely failed** (model ignored training, rambled about "looking for a job") |
| v2 | Added `c_fc`/`c_proj`, more escalation examples, 56 examples | Fixed refund escalation cleanly, but escalation language **bled into greetings** ("Hi there! I'm connecting you with our support team...") — a category-imbalance side effect of over-weighting refund examples |
| v3 | Added 6 pure-greeting examples with zero escalation language, 62 examples total | **All 4 test categories resolved correctly** — greetings stayed clean, refund escalation still worked, no regression |

### Final before/after comparison

| Prompt | Base GPT-2 (untrained) | LoRA fine-tuned |
|---|---|---|
| "how long does shipping take?" | Loops the question back endlessly | "Shipping takes about 3-5 business days depending on your order size." |
| "does this work for dandruff?" | Repeats "yes" in a loop | "Yes, it works great for dandruff too!..." (on-topic, coherent) |
| "I want a refund, this didn't work for me" | "I'm sorry, I didn't understand..." (loops) | "I'm sorry to hear that! I'm connecting you with our support team to sort this out." |
| "hi" | Rambles about being "a software engineer looking for a job" | "Hi there! 👋 How can I help you today?" |

---

## Key learnings

1. **LoRA's cost saving comes from never materializing the full-size update
   matrix** during training — only two small matrices (`A`, `B`) ever exist,
   multiplied sequentially through a narrow bottleneck of size `r`.
2. **Targeting only attention layers isn't enough** for tasks requiring content
   change (not just tone) — adding MLP layers (`c_fc`, `c_proj`) was necessary
   to fix the "hi" greeting failure.
3. **Fixing one category can break another.** Tripling refund/escalation
   examples fixed refunds but caused escalation language to bleed into
   greetings — a direct, hands-on lesson in category balance and the
   importance of evaluating *each* category, not just overall loss.
4. **Loss plateauing (~1.6-1.8) is expected and healthy**, not a failure —
   with 62 diverse examples across 5 categories, a small-rank adapter, and a
   decaying learning rate, the loss floor reflects genuine task diversity
   rather than an undertrained model. Driving it to near-zero would likely
   indicate overfitting/memorization instead.
5. **Loss curves alone don't tell you if fine-tuning worked** — the "hi"
   failure in v1 wasn't visible from the loss number, only from actually
   generating outputs and checking against a rubric per category.

## Limitations (honest caveats)

- GPT-2 (2019) is small and dated by modern LLM standards; a newer small model
  would likely need less data to generalize well.
- 62 examples is small for production use — some outputs (e.g. the dandruff
  answer inventing a plausible-sounding but unverified ingredient claim) show
  mild hallucination that more data or a retrieval-based guardrail would help fix.
- No proper held-out eval set was used — evaluation was 4 fixed prompts,
  checked manually. A real deployment would need a larger, systematic eval
  set and automated rubric scoring.
- Regulatory-sensitive claims (e.g. medical/cure claims) should not rely on
  fine-tuning alone in production — a rule-based or retrieval-based guardrail
  layer is recommended alongside the fine-tuned model, similar to how
  deterministic guardrails complement LLM judgment in other agentic systems.

## How to run

```bash
pip3 install torch transformers peft datasets accelerate

# Part 1: from-scratch LoRA demo
python3 train_demo.py

# Part 2: real GPT-2 + peft fine-tuning
python3 dataset.py          # verify dataset loads
python3 prepare_data.py     # verify tokenization + loss masking
python3 train_lora.py       # trains and saves adapter to ./lora_adapter
python3 test_model.py       # compares base vs. fine-tuned model
```

## File structure

```
lora_from_scratch/
├── lora.py           # hand-written LoRALinear + inject_lora()
├── tiny_gpt.py        # from-scratch decoder-only transformer
└── train_demo.py      # full fine-tune vs. LoRA comparison + merge check

lora_real_model/
├── dataset.py          # 62 customer-support Q&A pairs
├── prepare_data.py     # tokenization + loss masking (-100 labels)
├── train_lora.py       # peft LoraConfig + Hugging Face Trainer
├── test_model.py       # base vs. fine-tuned side-by-side comparison
└── lora_adapter/        # saved LoRA adapter weights (output)
```
