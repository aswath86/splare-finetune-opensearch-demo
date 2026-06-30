"""SageMaker inference handler for FROZEN SPLARE (no LoRA, no training).

Reads layer + L0 from env vars SPLARE_LAYER and SPLARE_L0. Uses base
Gemma-2-2B + frozen Gemma Scope SAE at the configured layer. This is the
sweep variant used to evaluate intrinsic retrieval quality per layer
without any task-specific training.
"""
import json, os, time, torch
from transformers import AutoModel, AutoTokenizer
from splare import SPLARE
from load_sae import load_sae

BACKBONE = "google/gemma-2-2b"
LAYER = int(os.environ.get("SPLARE_LAYER", "18"))
WIDTH = int(os.environ.get("SPLARE_WIDTH", "65"))  # in thousands
L0 = int(os.environ.get("SPLARE_L0", "116"))
MAX_LEN = int(os.environ.get("SPLARE_MAX_LEN", "512"))
SAE_REPO = os.environ.get("SPLARE_SAE_REPO", "google/gemma-scope-2b-pt-res")


def disable_causal_mask(llm):
    for m in llm.modules():
        if hasattr(m, "is_causal"):
            m.is_causal = False


def model_fn(model_dir):
    print(f"[model_fn] FROZEN mode: layer={LAYER} width={WIDTH}k L0={L0}", flush=True)
    W, b, th = load_sae(SAE_REPO, LAYER, WIDTH, L0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    llm = AutoModel.from_pretrained(BACKBONE, torch_dtype=dtype, attn_implementation="eager")
    disable_causal_mask(llm)  # bidirectional attention, matches v3 eval setup
    # NO LoRA load: this is the frozen variant
    W, b, th = W.to(device), b.to(device), th.to(device)
    splare = SPLARE(llm, W, b, layer_l=LAYER, sae_threshold=th).to(device).eval()
    print(f"[model_fn] ready on {device}", flush=True)
    return {"splare": splare, "tok": tok, "device": device}


def input_fn(body, content_type):
    if content_type != "application/json":
        raise ValueError(f"unsupported content_type {content_type}")
    return json.loads(body)


def predict_fn(data, ctx):
    texts = data.get("texts") or ([data["text"]] if "text" in data else [])
    # top_k: if >0, truncate to top-k features per row. 0 => return all non-zero.
    k = int(data.get("top_k", 0))
    if not texts:
        return {"response": [], "features": [], "latency_ms": 0.0}
    t0 = time.perf_counter()
    enc = ctx["tok"](texts, padding=True, truncation=True, max_length=MAX_LEN,
                    return_tensors="pt").to(ctx["device"])
    with torch.no_grad():
        u = ctx["splare"](enc["input_ids"], enc["attention_mask"])
    out = []
    for row in u:
        if k > 0:
            vals, idx = torch.topk(row, k)
            items = list(zip(idx.tolist(), vals.tolist()))
        else:
            nz = torch.nonzero(row, as_tuple=False).squeeze(-1)
            vals_nz = row[nz]
            items = list(zip(nz.tolist(), vals_nz.tolist()))
        pairs = {f"f_{i:05d}": round(float(v), 4) for i, v in items if v > 0}
        out.append(pairs)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {"response": out, "features": out, "latency_ms": round(dt_ms, 2),
            "batch_size": len(texts)}


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
