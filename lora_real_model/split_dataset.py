"""
Splits QA_PAIRS into a training set and a held-out validation set.
Validation examples are NEVER used during training — only for evaluation.
"""
import random
from dataset import QA_PAIRS

random.seed(42)  # fixed seed so the split is reproducible every time

def split_dataset(val_fraction=0.15):
    shuffled = QA_PAIRS.copy()
    random.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_fraction))
    val_set = shuffled[:val_size]
    train_set = shuffled[val_size:]

    return train_set, val_set


if __name__ == "__main__":
    train_set, val_set = split_dataset()
    print(f"Total examples: {len(QA_PAIRS)}")
    print(f"Train set: {len(train_set)}")
    print(f"Validation set (held out): {len(val_set)}")
    print("\n--- Validation examples (never trained on) ---")
    for i, ex in enumerate(val_set):
        print(f"{i+1}. Customer: {ex['customer']}")
        print(f"   Support:  {ex['support']}\n")
