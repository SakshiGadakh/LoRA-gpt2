"""
Converts training examples into tokenized examples with loss masking,
so the model only learns to predict the Support reply, not the Customer question.
Uses ONLY the training split — validation examples are held out entirely.
"""
from transformers import AutoTokenizer
from split_dataset import split_dataset

MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

MAX_LEN = 128


def build_example(pair):
    prompt = f"Customer: {pair['customer']}\nSupport:"
    full_text = f"{prompt} {pair['support']}{tokenizer.eos_token}"

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    full_ids = full_ids[:MAX_LEN]
    prompt_len = min(len(prompt_ids), len(full_ids))

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    labels = labels[:len(full_ids)]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def build_dataset():
    """Returns tokenized TRAINING examples only (validation set excluded)."""
    train_set, _ = split_dataset()
    return [build_example(p) for p in train_set]


if __name__ == "__main__":
    examples = build_dataset()
    print(f"Built {len(examples)} tokenized TRAINING examples (validation set excluded).\n")

    ex = examples[0]
    print("Example input_ids length:", len(ex["input_ids"]))
    print("Decoded full text:")
    print(tokenizer.decode(ex["input_ids"]))

    masked_count = sum(1 for l in ex["labels"] if l == -100)
    print(f"\nTokens masked (ignored in loss): {masked_count} / {len(ex['labels'])}")
