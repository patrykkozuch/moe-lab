import random
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from moe.models import SimpleMoETransformer
from moe.data import load_and_split_dataset, make_collate_fn
from moe.training import train_epoch, evaluate, predict


def main():
    # Hyperparameters
    SEED = 0
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model config
    MODEL_DIM = 128
    NUM_HEADS = 4
    FF_DIM = 512
    NUM_EXPERTS = 5
    NUM_LAYERS = 2
    DROPOUT = 0.1

    # Training config
    BATCH_SIZE = 256
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 4
    MAX_LENGTH = 192

    # Output
    CHECKPOINT_PATH = "simple_moe_transformer.pth"

    # Set seeds for reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Using device: {DEVICE}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)

    # Load and prepare datasets
    print("Loading datasets...")
    train_ds, val_ds = load_and_split_dataset()
    collate_fn = make_collate_fn(tokenizer, max_length=MAX_LENGTH)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize model
    print("Initializing model...")
    model = SimpleMoETransformer(
        vocab_size=tokenizer.vocab_size,
        dim=MODEL_DIM,
        heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_experts=NUM_EXPERTS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\nStarting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_epoch(model, optimizer, train_loader, DEVICE)
        acc = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch:02d} | loss {loss:.3f} | val_acc {acc:.3f}")

    # Test on sample texts
    print("\nTesting on sample texts:")
    samples = [
        "I really love this movie, it was fantastic!",
        "This was horrible and a waste of time."
    ]
    for text in samples:
        result = predict(model, text, tokenizer, DEVICE)
        print(f"{text}")
        print(f"  -> {result['label']} (prob={result['probability']:.3f})")

    # Save model
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"\nModel saved to {CHECKPOINT_PATH}")


if __name__ == '__main__':
    main()

