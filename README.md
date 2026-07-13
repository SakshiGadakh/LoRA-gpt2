# LoRA Fine-Tuning: GPT-2 Customer Support Chatbot

Fine-tuning **GPT-2 (124M params)** with **LoRA** (via Hugging Face's `peft`
library) to build a customer-support chatbot for a hair-care/skincare brand —
including dataset curation, hyperparameter tuning, a genuine train → eval →
diagnose → fix iteration cycle, and a quantitative evaluation on a held-out
validation set.

**Live demo:** https://huggingface.co/spaces/Sakshi12323/lora-support-chatbot-demo
**Adapter weights:** https://huggingface.co/Sakshi12323/lora-support-chatbot-gpt2

## Why this project

Fine-tuning APIs like [Tinker](https://tinker-docs.thinkingmachines.ai/) abstract
LoRA away behind a single config parameter (`rank=16`). This project applies
LoRA to a real pretrained model with a real use case, following a rigorous
methodology end to end: curate data, tune hyperparameters, train, and evaluate
quantitatively — rather than just running a fine-tuning script once and
calling it done.

---

## The project

**Files:** `lora_real_model/dataset.py`, `lora_real_model/split_dataset.py`, `lora_real_model/prepare_data.py`, `lora_real_model/train_lora.py`, `lora_real_model/test_model.py`, `lora_real_model/evaluate.py`

Applied LoRA to **GPT-2 (124M params)** to build a customer-support chatbot for
a generic hair-care/skincare brand, using Hugging Face's `peft` library.

### 1. Dataset curation

82 hand-written customer-support Q&A pairs across 5 categories: product usage,
shipping, returns/refunds (including escalation-sensitive cases like medical
claims), complaints/objections, and greetings/general tone. Each example is
tokenized with **loss masking** — the model only trains on the "Support:"
reply tokens, not the "Customer:" question, using `-100` labels for the
ignored portion (matching the same `Datum` concept used in Tinker's SFT API).

The dataset went through two rounds of targeted expansion, driven by observed
failures rather than guesswork (see "Iteration history" below).

### 2. Train/validation split

`split_dataset.py` holds out **12 examples (15%) as a validation set** with a
fixed random seed, ensuring these examples are **never seen during training**
— only used afterward for honest evaluation. The remaining 70 examples form
the training set.

### 3. LoRA hyperparameters (final version)

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_fc", "c_proj"],  # attention + MLP layers
)
```

- **Trainable params:** 2,359,296 / 126,799,104 total (**1.86%**)
- Rank and target modules were chosen empirically, based on diagnosing why an
  earlier attention-only configuration failed to shift model *content* (not
  just tone) — see "Iteration history."

### 4. Training / epochs

12 epochs, batch size 4, learning rate 2e-4, trained on CPU/MPS in ~20-26
seconds. Epoch count was reduced from an earlier 15-epoch run once more
training data was added, to reduce overfitting risk on any single example.

### Iteration history — a real train → eval → diagnose → fix loop

| Version | Change | Outcome |
|---|---|---|
| v1 | `target_modules=["c_attn"]` only, 36 examples | Shipping/refund/dandruff worked partially; **greetings completely failed** (model ignored training, rambled about "looking for a job") |
| v2 | Added `c_fc`/`c_proj`, more escalation examples, 56 examples | Fixed refund escalation cleanly, but escalation language **bled into greetings** ("Hi there! I'm connecting you with our support team...") — a category-imbalance side effect of over-weighting refund examples |
| v3 | Added 6 pure-greeting examples with zero escalation language, 62 examples total | All 4 spot-check categories resolved correctly — greetings stayed clean, refund escalation still worked, no regression |
| v4 | Expanded to 82 examples, added a proper 70/12 train/validation split, retrained | Enabled genuine quantitative evaluation on held-out data (see below) |

### 5. Evaluation on held-out validation data

Ran the fine-tuned model and the base model on all **12 validation examples**
(never seen during training), and scored each output against the dataset's
reference answer using **ROUGE-1**, **ROUGE-L** (word/phrase overlap), and
**BERTScore** (semantic similarity via `roberta-large` embeddings).

| Metric | Base GPT-2 | LoRA fine-tuned | Change |
|---|---|---|---|
| ROUGE-1 | 0.097 | 0.181 | +87% |
| ROUGE-L | 0.093 | 0.156 | +68% |
| BERTScore F1 | 0.841 | 0.876 | +4.2% |

LoRA fine-tuning improved over the base model on **every metric**, on data the
model never trained on — this is the key result that distinguishes genuine
generalization from memorization.

**An honest limitation, found during evaluation itself:** BERTScore's gain
(+4.2%) was much smaller than ROUGE's (+68–87%). Digging into individual
examples explains why — a few LoRA outputs were fluent, on-topic, and
structurally close to real support replies (driving ROUGE up), but were
**factually backwards or hallucinated**: e.g. answering "yes, we sell
perfumes" when the brand doesn't, confirming "yes, this is a scam" to a
customer asking if the brand was legitimate, and inventing a specific but
fabricated delivery-scheduling policy. Neither ROUGE nor BERTScore penalize
this kind of confident, fluent factual reversal — both measure textual/semantic
similarity, not correctness. This is exactly the gap that an LLM-as-judge
evaluation or a rule-based safety check would close, and it's a genuine
limitation of this evaluation, not just of the model.

---

## Key learnings

1. **Targeting only attention layers isn't enough** for tasks requiring content
   change (not just tone) — adding MLP layers (`c_fc`, `c_proj`) was necessary
   to fix a "hi" greeting failure where the model ignored training entirely.
2. **Fixing one category can break another.** Tripling refund/escalation
   examples fixed refunds but caused escalation language to bleed into
   greetings — a direct, hands-on lesson in category balance and the
   importance of evaluating *each* category, not just overall loss.
3. **Loss plateauing (~1.6-1.8) is expected and healthy**, not a failure —
   with diverse examples across 5 categories, a small-rank adapter, and a
   decaying learning rate, the loss floor reflects genuine task diversity
   rather than an undertrained model. Driving it to near-zero would likely
   indicate overfitting/memorization instead.
4. **Loss curves alone don't tell you if fine-tuning worked** — a greeting
   failure in an early version wasn't visible from the loss number, only from
   actually generating outputs and checking against a rubric per category.
5. **Word-overlap and embedding-similarity metrics can both be fooled by
   fluent, confidently wrong answers.** A model can score well on ROUGE and
   BERTScore while stating something factually backwards — quantitative NLP
   metrics measure similarity to a reference, not truthfulness, and should be
   paired with targeted checks for known high-risk failure modes.

## Limitations (honest caveats)

- GPT-2 (2019) is small and dated by modern LLM standards; a newer small model
  would likely need less data to generalize well.
- 82 examples (70 train / 12 validation) is small for production use — several
  outputs show hallucinated or factually reversed claims (see Evaluation
  section) that more data, better data balance, or a retrieval-based guardrail
  would help address.
- ROUGE and BERTScore, the metrics used here, do not catch factual
  correctness or contradiction — an LLM-as-judge evaluation or a rule-based
  fact/safety checker would be needed to close this gap.
- Regulatory-sensitive claims (e.g. medical/cure claims) should not rely on
  fine-tuning alone in production — a rule-based or retrieval-based guardrail
  layer is recommended alongside the fine-tuned model, similar to how
  deterministic guardrails complement LLM judgment in other agentic systems.

## How to run

```bash
pip3 install torch transformers peft datasets accelerate rouge-score bert-score

cd lora_real_model
python3 dataset.py          # verify full dataset loads (82 examples)
python3 split_dataset.py    # view the 70/12 train/validation split
python3 prepare_data.py     # verify tokenization + loss masking (train set only)
python3 train_lora.py       # trains on the 70 training examples, saves adapter
python3 test_model.py       # quick base vs. fine-tuned spot check (4 prompts)
python3 evaluate.py         # full ROUGE + BERTScore evaluation on held-out set
```

## Try the live demo

A hosted Gradio demo comparing base GPT-2 vs. the LoRA fine-tuned model is
available at:

**https://huggingface.co/spaces/Sakshi12323/lora-support-chatbot-demo**

Demo source code: `hf_space/app.py`

## File structure

```
LoRA-gpt2/
├── lora_real_model/
│   ├── dataset.py           # 82 customer-support Q&A pairs
│   ├── split_dataset.py     # 70/12 train/validation split (fixed seed)
│   ├── prepare_data.py      # tokenization + loss masking (-100 labels), train set only
│   ├── train_lora.py        # peft LoraConfig + Hugging Face Trainer
│   ├── test_model.py        # base vs. fine-tuned quick spot check
│   ├── evaluate.py          # ROUGE + BERTScore evaluation on held-out validation set
│   └── lora_adapter/        # saved LoRA adapter weights (output, gitignored)
└── hf_space/
    ├── app.py                # Gradio demo app (deployed to Hugging Face Spaces)
    └── requirements.txt
```
