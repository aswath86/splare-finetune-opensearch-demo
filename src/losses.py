"""
Losses for SPLARE training.

kl_distillation_loss: paper eq. (3) — KL from teacher softmax to student softmax
flops_loss:           paper eq. (4) tail — sum((mean_over_batch |x|)^2)
"""
import torch
import torch.nn.functional as F


def kl_distillation_loss(student_scores: torch.Tensor,
                         teacher_scores: torch.Tensor,
                         temperature: float) -> torch.Tensor:
    """
    student_scores: (B, 1+n_neg)  — sparse dot products q·d
    teacher_scores: (B, 1+n_neg)  — cross-encoder / BGE reranker scores
    Returns scalar KL(teacher || student).
    """
    p = F.softmax(teacher_scores, dim=-1)
    log_q = F.log_softmax(student_scores / temperature, dim=-1)
    log_p = F.log_softmax(teacher_scores, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1).mean()


def flops_loss(rep: torch.Tensor) -> torch.Tensor:
    """rep: (B, |W|) sparse vectors. Returns scalar FLOPS penalty."""
    return (rep.abs().mean(dim=0) ** 2).sum()
