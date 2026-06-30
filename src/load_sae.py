"""
Download a Gemma Scope JumpReLU SAE and return (W_enc, b_enc, threshold) as torch tensors.

Gemma Scope file layout:
  google/gemma-scope-2b-pt-res/layer_{L}/width_{W}k/average_l0_{N}/params.npz
Keys inside params.npz: W_enc, W_dec, b_enc, b_dec, threshold
  W_enc shape: (d_model, |W|)   so our module needs W_enc.T -> (|W|, d_model)
  b_enc shape: (|W|,)
  threshold shape: (|W|,)       per-feature JumpReLU threshold
"""
import argparse
import numpy as np
import torch
from huggingface_hub import hf_hub_download


def load_sae(repo_id: str, layer: int, width: int, l0: int):
    """width in thousands (e.g. 65 for width_65k), l0 is the average_l0_N variant."""
    width_str = f"width_{width}k"
    path = hf_hub_download(
        repo_id=repo_id,
        filename=f"layer_{layer}/{width_str}/average_l0_{l0}/params.npz",
    )
    npz = np.load(path)
    W_enc = torch.from_numpy(npz["W_enc"]).float().T.contiguous()   # (|W|, d)
    b_enc = torch.from_numpy(npz["b_enc"]).float()                  # (|W|,)
    threshold = torch.from_numpy(npz["threshold"]).float()          # (|W|,)
    return W_enc, b_enc, threshold


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="google/gemma-scope-2b-pt-res")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--width", type=int, default=65, help="in thousands")
    p.add_argument("--l0", type=int, default=107)
    args = p.parse_args()

    W, b, th = load_sae(args.repo, args.layer, args.width, args.l0)
    print(f"W_enc: {tuple(W.shape)}  b_enc: {tuple(b.shape)}  threshold: {tuple(th.shape)}")
    print(f"threshold stats: min={th.min():.4f}  median={th.median():.4f}  max={th.max():.4f}")
