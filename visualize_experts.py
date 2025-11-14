import os
from typing import List

import torch
from transformers import AutoTokenizer

from moe.models import SimpleMoETransformer


def load_model(weights_path: str, device: torch.device):
    """
    Load a trained SimpleMoE model and tokenizer.

    Args:
        weights_path: Path to saved model weights
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
    model = SimpleMoETransformer(vocab_size=tokenizer.vocab_size).to(device)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights file not found: {weights_path}\n"
            f"Train a model first using: python train.py"
        )

    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


def visualize_expert_probs(
    probs: torch.Tensor,
    tokens: List[str],
    layer_idx: int,
    out_path: str,
    show: bool = False
):
    """
    Visualize expert probabilities for a single layer as a heatmap.

    Args:
        probs: Expert probabilities tensor (seq_len, num_experts)
        tokens: List of token strings (length seq_len)
        layer_idx: Layer index for title
        out_path: Path to save the figure
        show: Whether to display the plot interactively
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for visualization.\n"
            "Install with: pip install matplotlib"
        )

    seq_len, num_experts = probs.shape

    # Dynamic figure sizing
    fig_w = float(max(8.0, min(0.4 * seq_len, 24.0)))
    fig_h = float(max(3.0, min(0.4 * num_experts + 1.5, 12.0)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(probs.T.cpu(), aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)

    ax.set_title(f"Expert Probabilities per Token (Layer {layer_idx})", fontsize=12, fontweight='bold')
    ax.set_ylabel("Experts", fontsize=10)
    ax.set_xlabel("Tokens", fontsize=10)

    # Expert labels on y-axis
    ax.set_yticks(list(range(num_experts)))
    ax.set_yticklabels([f"E{i}" for i in range(num_experts)])

    # Token labels on x-axis (truncate long tokens)
    ax.set_xticks(list(range(seq_len)))
    short_tokens = [t if len(t) <= 15 else t[:12] + '…' for t in tokens]
    ax.set_xticklabels(short_tokens, rotation=45, ha='right', fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Probability', fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Saved heatmap to: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def visualize_all_expert_probs(
    probs_list: List[torch.Tensor],
    tokens: List[str],
    out_path: str,
    show: bool = False
):
    """
    Visualize expert probabilities for all layers in a single figure.

    Args:
        probs_list: List of probability tensors, one per layer (seq_len, num_experts)
        tokens: List of token strings (length seq_len)
        out_path: Path to save the figure
        show: Whether to display the plot interactively
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Matplotlib is required for visualization.\n"
            "Install with: pip install matplotlib"
        )

    num_layers = len(probs_list)
    if num_layers == 0:
        raise ValueError("No probability tensors provided")

    seq_len, num_experts = probs_list[0].shape

    # Dynamic figure sizing
    fig_w = float(max(8.0, min(0.4 * seq_len, 28.0)))
    per_layer_h = float(max(2.5, min(0.25 * num_experts + 0.8, 6.0)))
    fig_h = float(min(36.0, num_layers * per_layer_h))

    fig, axes = plt.subplots(nrows=num_layers, ncols=1, figsize=(fig_w, fig_h), sharex=True)
    if num_layers == 1:
        axes = [axes]

    ims = []
    for layer_idx, (ax, probs) in enumerate(zip(axes, probs_list)):
        im = ax.imshow(probs.T, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
        ims.append(im)
        ax.set_ylabel(f"L{layer_idx}\nExperts", fontsize=9)
        ax.set_yticks(list(range(num_experts)))
        ax.set_yticklabels([f"E{i}" for i in range(num_experts)], fontsize=8)
        ax.set_title(f"Layer {layer_idx}", fontsize=10, fontweight='bold')

    axes[-1].set_xlabel("Tokens", fontsize=10)
    axes[-1].set_xticks(list(range(seq_len)))
    short_tokens = [t if len(t) <= 15 else t[:12] + '…' for t in tokens]
    axes[-1].set_xticklabels(short_tokens, rotation=45, ha='right', fontsize=8)

    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"Saved combined heatmap to: {out_path}")

    if show:
        plt.show()
    plt.close(fig)


def visualize_text(
    text: str,
    weights_path: str = "simple_moe_transformer.pth",
    device: str = "cpu",
    layer: int = 0,
    all_layers: bool = False,
    max_tokens: int = 64,
    output_path: str = None,
    show: bool = False
):
    """
    Visualize expert activations for a given text.

    Args:
        text: Input text to analyze
        weights_path: Path to model weights
        device: Device to run on ('cpu' or 'cuda')
        layer: Which layer to visualize (ignored if all_layers=True)
        all_layers: Whether to visualize all layers
        max_tokens: Maximum number of tokens to display
        output_path: Path to save figure (auto-generated if None)
        show: Whether to display plot interactively
    """
    device = torch.device(device)
    model, tokenizer = load_model(weights_path, device)

    # Tokenize input
    enc = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    x = input_ids.to(device)
    m = attn_mask.to(torch.float32).to(device)

    # Run model
    with torch.no_grad():
        logits, probs_layers = model.forward_with_expert_probs(x, m)
        prob = torch.sigmoid(logits[0]).item()
        pred = int(prob > 0.5)
        print(f"Text: {text}")
        print(f"Prediction: {'POS' if pred == 1 else 'NEG'} (probability={prob:.3f})")

    # Extract valid token positions (non-padding)
    mask_list = attn_mask[0].tolist()
    valid_positions = [i for i, mv in enumerate(mask_list) if mv == 1]
    if max_tokens and max_tokens > 0:
        valid_positions = valid_positions[:max_tokens]

    # Get tokens and probabilities for valid positions
    token_ids = [input_ids[0, i].item() for i in valid_positions]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Slice probabilities to valid positions
    sliced_probs = [pl[0][valid_positions].cpu() for pl in probs_layers]

    # Generate output path if not provided
    if output_path is None:
        if all_layers:
            output_path = "expert_heatmap_all_layers.png"
        else:
            output_path = f"expert_heatmap_layer{layer}.png"

    # Visualize
    if all_layers:
        visualize_all_expert_probs(sliced_probs, tokens, output_path, show)
    else:
        if layer < 0 or layer >= len(sliced_probs):
            print(f"Warning: Layer {layer} out of range [0, {len(sliced_probs)-1}], using layer 0")
            layer = 0
        visualize_expert_probs(sliced_probs[layer], tokens, layer, output_path, show)


if __name__ == "__main__":
    # Example usage
    visualize_text(
        text="I watched The Great Escape last night and it was a total surprise. The stunt work was absolutely incredible, and the car chases had me on the edge of my seat the whole time. If you want a fun, action-packed movie that doesn't make you think too hard, go see this right now.",
        all_layers=True,
        show=False
    )

