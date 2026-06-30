"""
SPLARE module: frozen SAE encoder on top of a decoder LLM hidden state, SPLADE-pooled.

Paper: "Learning Retrieval Models with Sparse Autoencoders" (arXiv:2603.13277)
Default config targets SPLARE-2B (Gemma-2-2B + Gemma Scope residual SAE at layer 6).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPLARE(nn.Module):
    def __init__(self, llm: nn.Module, sae_W_enc: torch.Tensor, sae_b_enc: torch.Tensor,
                 layer_l: int, sae_threshold: torch.Tensor | None = None):
        """
        llm: HF causal/masked LM with output_hidden_states support. Apply LoRA externally.
        sae_W_enc: (|W|, d)  SAE encoder weight (frozen).
        sae_b_enc: (|W|,)    SAE encoder bias (frozen).
        layer_l: index into hidden_states (1..num_layers; 0 is embeddings).
        sae_threshold: (|W|,) optional JumpReLU threshold; if None, plain ReLU is used.
        """
        super().__init__()
        self.llm = llm
        self.layer_l = layer_l
        self.register_buffer("W_enc", sae_W_enc, persistent=False)
        self.register_buffer("b_enc", sae_b_enc, persistent=False)
        if sae_threshold is not None:
            self.register_buffer("threshold", sae_threshold, persistent=False)
        else:
            self.threshold = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns SPLADE-pooled sparse vector of shape (B, |W|)."""
        out = self.llm(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        h = out.hidden_states[self.layer_l]                   # (B, N, d)
        # Match dtype of backbone hidden state (bf16/fp16/fp32) for matmul compat
        W = self.W_enc.to(h.dtype)
        b = self.b_enc.to(h.dtype)
        pre = h @ W.T + b                                     # (B, N, |W|)
        if self.threshold is not None:
            th = self.threshold.to(h.dtype)
            z = F.relu(pre) * (pre > th)                      # JumpReLU
        else:
            z = F.relu(pre)
        mask = attention_mask.unsqueeze(-1).bool()
        u = torch.log1p(z).masked_fill(~mask, 0.0).max(dim=1).values
        return u                                              # (B, |W|)


def top_k_prune(u: torch.Tensor, k: int) -> torch.Tensor:
    """Inference-time Top-K pooling; zeros out all but top-k per row."""
    if k <= 0 or k >= u.shape[-1]:
        return u
    vals, idx = torch.topk(u, k, dim=-1)
    out = torch.zeros_like(u)
    return out.scatter_(-1, idx, vals)
