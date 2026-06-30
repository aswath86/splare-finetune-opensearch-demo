"""
Single training step computation. Given a batch of (query, pos, negs) samples
and their teacher scores, compute SPLARE representations and the total loss.
"""
import torch
from losses import kl_distillation_loss, flops_loss


def splare_step(splare, tok, batch, max_len, temperature, lambda_q, lambda_d, device):
    """
    splare: SPLARE module (encode text -> (B, |W|) sparse vector)
    tok:    HF tokenizer
    batch:  list of dicts {query, pos, negs (list len n), pos_score, neg_scores}
    Returns scalar loss tensor + diagnostics dict.
    """
    n_neg = len(batch[0]["negs"])
    queries = [b["query"] for b in batch]
    pos     = [b["pos"]   for b in batch]
    negs_flat = [n for b in batch for n in b["negs"]]   # (B*n_neg,)

    q_ids  = tok(queries, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    d_ids  = tok(pos + negs_flat, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)

    q_rep = splare(q_ids["input_ids"], q_ids["attention_mask"])              # (B, |W|)
    d_rep = splare(d_ids["input_ids"], d_ids["attention_mask"])              # (B + B*n_neg, |W|)

    B = len(batch)
    pos_rep  = d_rep[:B]                      # (B, |W|)
    neg_rep  = d_rep[B:].view(B, n_neg, -1)   # (B, n_neg, |W|)

    pos_score = (q_rep * pos_rep).sum(-1, keepdim=True)            # (B, 1)
    neg_score = torch.einsum("bv,bnv->bn", q_rep, neg_rep)         # (B, n_neg)
    student = torch.cat([pos_score, neg_score], dim=-1)            # (B, 1+n_neg)

    teacher = torch.tensor(
        [[b["pos_score"]] + b["neg_scores"] for b in batch],
        dtype=student.dtype, device=device
    )                                                              # (B, 1+n_neg)

    l_kl  = kl_distillation_loss(student, teacher, temperature=temperature)
    l_q   = flops_loss(q_rep)
    l_d   = flops_loss(torch.cat([pos_rep, neg_rep.reshape(-1, pos_rep.shape[-1])], dim=0))
    loss  = l_kl + lambda_q * l_q + lambda_d * l_d

    diag = {
        "loss": loss.item(),
        "kl":   l_kl.item(),
        "flops_q": l_q.item(),
        "flops_d": l_d.item(),
        "nnz_q_mean": (q_rep > 0).sum(-1).float().mean().item(),
        "nnz_d_mean": (d_rep > 0).sum(-1).float().mean().item(),
    }
    return loss, diag
