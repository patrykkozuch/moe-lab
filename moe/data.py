import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_and_split_dataset(dataset_name: str = "stanfordnlp/imdb", val_ratio: float = 0.2, seed: int = 0):
    """
    Load and split a dataset into train and validation sets.

    Args:
        dataset_name: HuggingFace dataset name
        val_ratio: Fraction of training data to use for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    raw = load_dataset(dataset_name)
    split = raw["train"].train_test_split(test_size=val_ratio, seed=seed)
    train_ds = split["train"].shuffle(seed=seed)
    val_ds = split["test"].shuffle(seed=seed)
    return train_ds, val_ds


def make_collate_fn(tokenizer: PreTrainedTokenizerBase, max_length: int = 192):
    """
    Create a collate function for batching text data.

    Args:
        tokenizer: Tokenizer to use for encoding text
        max_length: Maximum sequence length

    Returns:
        Collate function for DataLoader
    """
    def collate(batch):
        texts = [item["text"] for item in batch]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"].to(torch.float32)
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return input_ids, attention_mask, labels

    return collate

