import torch
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader


def train_epoch(model, optimizer, loader: DataLoader, device: torch.device) -> float:
    """
    Train the model for one epoch.

    Args:
        model: Model to train
        optimizer: Optimizer
        loader: Training data loader
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for input_ids, attention_mask, labels in tqdm.tqdm(loader, desc="Training"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device).float()

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        total_samples += input_ids.size(0)

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate model accuracy on a dataset.

    Args:
        model: Model to evaluate
        loader: Validation data loader
        device: Device to evaluate on

    Returns:
        Accuracy as a fraction
    """
    model.eval()
    correct = 0
    total = 0

    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids, attention_mask)
        predictions = (torch.sigmoid(logits) > 0.5).long()
        correct += torch.eq(predictions, labels).sum().item()
        total += labels.size(0)

    return correct / max(total, 1)


@torch.no_grad()
def predict(model, text: str, tokenizer, device: torch.device) -> dict:
    """
    Make a prediction on a single text sample.

    Args:
        model: Model to use for prediction
        text: Input text
        tokenizer: Tokenizer
        device: Device to run on

    Returns:
        Dictionary with 'probability', 'prediction', and 'label'
    """
    model.eval()

    enc = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(torch.float32).to(device)

    logit = model(input_ids, attention_mask)
    prob = torch.sigmoid(logit[0]).item()
    pred = int(prob > 0.5)

    return {
        'probability': prob,
        'prediction': pred,
        'label': 'POS' if pred == 1 else 'NEG'
    }

