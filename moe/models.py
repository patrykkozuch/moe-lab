import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SimpleMoE(nn.Module):
    """
    Simple MoE layer with top-k=2 routing (soft mixture of top 2 experts).
    
    Args:
        dim: Hidden dimension size
        ff_dim: Feed-forward dimension size
        num_experts: Number of expert networks
        dropout: Dropout probability
    """
    def __init__(self, dim: int, ff_dim: int, num_experts: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)
        self.dropout = nn.Dropout(dropout)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, ff_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, dim),
                nn.Dropout(dropout),
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with expert routing.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        logits = self.gate(x)  # (b, s, E)
        probs = F.softmax(logits, dim=-1)  # (b, s, E)
        all_out = torch.stack([e(x) for e in self.experts], dim=2)  # (b, s, E, d)
        top_vals, top_idx = probs.topk(2, dim=-1)  # (b, s, 2)

        weights = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-9)  # (b, s, 2)

        selected = all_out.gather(
            2,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, x.size(-1))
        )  # (b, s, 2, d)
        gathered = (selected * weights.unsqueeze(-1)).sum(dim=2)  # (b, s, d)
        return self.dropout(gathered)

    def forward_with_probs(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns expert probability distribution per token.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            
        Returns:
            Tuple of (output, probs) where:
                - output: (batch, seq_len, dim)
                - probs: Expert probabilities (batch, seq_len, num_experts)
        """
        logits = self.gate(x)  # (b, s, E)
        probs = F.softmax(logits, dim=-1)  # (b, s, E)
        all_out = torch.stack([e(x) for e in self.experts], dim=2)  # (b, s, E, d)
        top_vals, top_idx = probs.topk(2, dim=-1)  # (b, s, 2)
        weights = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-9)  # (b, s, 2)
        selected = all_out.gather(
            2,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, x.size(-1))
        )  # (b, s, 2, d)
        gathered = (selected * weights.unsqueeze(-1)).sum(dim=2)  # (b, s, d)
        return self.dropout(gathered), probs


class SimpleMoEDecoderLayer(nn.Module):
    """
    Transformer decoder layer with MoE feed-forward.
    
    Args:
        dim: Hidden dimension size
        heads: Number of attention heads
        ff_dim: Feed-forward dimension size
        num_experts: Number of expert networks
        dropout: Dropout probability
    """
    def __init__(self, dim: int, heads: int, ff_dim: int, num_experts: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.moe = SimpleMoE(dim, ff_dim, num_experts, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder layer.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Attention mask (batch, seq_len)
            
        Returns:
            Output tensor (batch, seq_len, dim)
        """
        key_padding = mask == 0
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding)
        x = self.ln1(x + self.dropout(attn_out))
        ff = self.moe(x)
        x = self.ln2(x + self.dropout(ff))
        return x

    def forward_with_probs(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns expert probabilities.
        
        Args:
            x: Input tensor (batch, seq_len, dim)
            mask: Attention mask (batch, seq_len)
            
        Returns:
            Tuple of (output, probs) where:
                - output: (batch, seq_len, dim)
                - probs: Expert probabilities (batch, seq_len, num_experts)
        """
        key_padding = mask == 0
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding)
        x = self.ln1(x + self.dropout(attn_out))
        ff, probs = self.moe.forward_with_probs(x)
        x = self.ln2(x + self.dropout(ff))
        return x, probs


class SimpleMoETransformer(nn.Module):
    """
    Complete MoE Transformer for binary classification.
    
    Args:
        vocab_size: Vocabulary size
        dim: Hidden dimension size
        heads: Number of attention heads
        ff_dim: Feed-forward dimension size
        num_experts: Number of expert networks per layer
        num_layers: Number of transformer layers
        dropout: Dropout probability
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        heads: int = 4,
        ff_dim: int = 512,
        num_experts: int = 5,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SimpleMoEDecoderLayer(dim, heads, ff_dim, num_experts, dropout) 
            for _ in range(num_layers)
        ])

        self.ln = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, 1)  # Single output for binary classification

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for binary classification.
        
        Args:
            x: Input token IDs (batch, seq_len)
            mask: Attention mask (batch, seq_len)
            
        Returns:
            Logits for binary classification (batch,)
        """
        x = self.dropout(self.embed(x))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln(x)
        cls_token = x[:, 0, :]
        logits = self.cls(cls_token).squeeze(-1)
        return logits

    def forward_with_expert_probs(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass that also returns expert probability distributions.
        
        Args:
            x: Input token IDs (batch, seq_len)
            mask: Attention mask (batch, seq_len)
            
        Returns:
            Tuple of (logits, probs_per_layer) where:
                - logits: Binary classification logits (batch,)
                - probs_per_layer: List of expert probabilities per layer
                  Each element has shape (batch, seq_len, num_experts)
        """
        probs_per_layer = []
        x = self.dropout(self.embed(x))
        for layer in self.layers:
            x, probs = layer.forward_with_probs(x, mask)
            probs_per_layer.append(probs)
        x = self.ln(x)
        cls_token = x[:, 0, :]
        logits = self.cls(cls_token).squeeze(-1)
        return logits, probs_per_layer

