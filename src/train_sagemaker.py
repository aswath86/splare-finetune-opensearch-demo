"""
SageMaker training entry point for SPLARE-2B LoRA fine-tune.

v2: layer 18 + bidirectional attention. No MNTP for now.
"""
import argparse, json, os, random, time
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from splare import SPLARE
from load_sae import load_sae
from step import splare_step


class JsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path, n_negs):
        items = [json.loads(l) for l in open(path)]
        for it in items:
            it["negs"] = it["negs"][:n_negs]
            it["neg_scores"] = it["neg_scores"][:n_negs]
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def disable_causal_mask(llm):
    """Force Gemma attention modules to be bidirectional."""
    n = 0
    for m in llm.modules():
        if hasattr(m, "is_causal"):
            m.is_causal = False
            n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-len", type=int, default=192)
    p.add_argument("--n-negs", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--temperature", type=float, default=50.0)
    p.add_argument("--lambda-q", type=float, default=1e-4)
    p.add_argument("--lambda-d", type=float, default=1e-4)
    p.add_argument("--sae-layer", type=int, default=18)
    p.add_argument("--sae-width", type=int, default=65)
    p.add_argument("--sae-l0", type=int, default=116)
    p.add_argument("--backbone", default="google/gemma-2-2b")
    p.add_argument("--bidirectional", type=int, default=1)
    p.add_argument("--mntp-steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for LoRA init + data shuffle. Vary for model averaging.")
    p.add_argument("--train-dir", default=os.environ.get("SM_CHANNEL_TRAIN", "data"))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "./output"))
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    random.seed(args.seed); torch.manual_seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"seed={args.seed}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} torch={torch.__version__}", flush=True)

    train_files = [f for f in os.listdir(args.train_dir) if f.endswith(".jsonl")]
    assert train_files, f"No .jsonl in {args.train_dir}"
    train_path = os.path.join(args.train_dir, train_files[0])
    print(f"train file: {train_path}", flush=True)

    print("Loading SAE (Gemma Scope)...", flush=True)
    W, b, th = load_sae("google/gemma-scope-2b-pt-res",
                        args.sae_layer, args.sae_width, args.sae_l0)
    W, b, th = W.to(device), b.to(device), th.to(device)

    print(f"Loading {args.backbone}...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.backbone)
    llm = AutoModel.from_pretrained(args.backbone, torch_dtype=torch.bfloat16,
                                     attn_implementation="eager").to(device)

    if args.bidirectional:
        n_patched = disable_causal_mask(llm)
        print(f"Bidirectional attention: patched {n_patched} attn modules", flush=True)

    lora_cfg = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank * 2,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          lora_dropout=0.0, bias="none", task_type="FEATURE_EXTRACTION")
    llm = get_peft_model(llm, lora_cfg)
    llm.print_trainable_parameters()

    # Optional: initialize LoRA adapter from MNTP artifact
    mntp_artifact = os.environ.get("MNTP_ARTIFACT")
    if mntp_artifact:
        import subprocess, tarfile
        local_tar = "/tmp/mntp.tar.gz"; local_dir = "/tmp/mntp_adapter"
        os.makedirs(local_dir, exist_ok=True)
        print(f"Downloading MNTP artifact {mntp_artifact}...", flush=True)
        subprocess.run(["aws", "s3", "cp", mntp_artifact, local_tar, "--only-show-errors"], check=True)
        with tarfile.open(local_tar) as t:
            t.extractall(local_dir)
        print(f"MNTP dir: {os.listdir(local_dir)}", flush=True)
        # Load LoRA state_dict from MNTP; names have CAUSAL_LM task_type but same target modules
        from safetensors.torch import load_file
        adapter_path = os.path.join(local_dir, "adapter_model.safetensors")
        mntp_state = load_file(adapter_path)
        current_names = dict(llm.named_parameters())
        loaded = 0; skipped = 0
        for name, val in mntp_state.items():
            # MNTP (CausalLM+PEFT): base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
            # SPLARE (AutoModel+PEFT): base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
            mapped = name.replace("base_model.model.model.", "base_model.model.", 1)
            mapped = mapped.replace(".lora_A.weight", ".lora_A.default.weight")
            mapped = mapped.replace(".lora_B.weight", ".lora_B.default.weight")
            if mapped in current_names and current_names[mapped].shape == val.shape:
                current_names[mapped].data.copy_(val.to(current_names[mapped].dtype).to(current_names[mapped].device))
                loaded += 1
            else:
                skipped += 1
        print(f"MNTP adapter: loaded {loaded} params, skipped {skipped}", flush=True)
        assert loaded > 0, "MNTP load produced zero matches — name mapping bug"

    if hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable()
    if hasattr(llm, "enable_input_require_grads"):
        llm.enable_input_require_grads()

    splare = SPLARE(llm, W, b, layer_l=args.sae_layer, sae_threshold=th).to(device)

    ds = JsonlDataset(train_path, n_negs=args.n_negs)
    print(f"train samples: {len(ds)}", flush=True)
    # Explicit seeded generator for DataLoader shuffle, so seed actually controls order
    loader_gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: b, num_workers=2, generator=loader_gen)

    opt = torch.optim.AdamW([p for p in splare.parameters() if p.requires_grad], lr=args.lr)
    splare.train()
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            t0 = time.time()
            loss, diag = splare_step(splare, tok, batch,
                                     max_len=args.max_len, temperature=args.temperature,
                                     lambda_q=args.lambda_q, lambda_d=args.lambda_d, device=device)
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad()
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step} [{time.time()-t0:.2f}s] "
                      f"loss={diag['loss']:.3f} kl={diag['kl']:.3f} "
                      f"flops_q={diag['flops_q']:.1f} flops_d={diag['flops_d']:.1f} "
                      f"nnz_q={diag['nnz_q_mean']:.0f} nnz_d={diag['nnz_d_mean']:.0f}",
                      flush=True)
            step += 1

    os.makedirs(args.model_dir, exist_ok=True)
    splare.llm.save_pretrained(args.model_dir)
    tok.save_pretrained(args.model_dir)
    with open(os.path.join(args.model_dir, "splare_config.json"), "w") as f:
        json.dump({
            "backbone": args.backbone,
            "sae_repo": "google/gemma-scope-2b-pt-res",
            "sae_layer": args.sae_layer,
            "sae_width": args.sae_width,
            "sae_l0": args.sae_l0,
            "bidirectional": bool(args.bidirectional),
        }, f, indent=2)
    print(f"Saved to {args.model_dir}", flush=True)


if __name__ == "__main__":
    main()
