"""SageMaker inference handler for SPLARE."""
import json, os, torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
from splare import SPLARE
from load_sae import load_sae

BACKBONE = "google/gemma-2-2b"
LAYER = 18
WIDTH = 65
L0 = 116
MAX_LEN = 1024
TOP_K = 40


def disable_causal_mask(llm):
    for m in llm.modules():
        if hasattr(m, "is_causal"):
            m.is_causal = False


def model_fn(model_dir):
    print(f"[model_fn] loading from {model_dir}", flush=True)
    W, b, th = load_sae("google/gemma-scope-2b-pt-res", LAYER, WIDTH, L0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    llm = AutoModel.from_pretrained(BACKBONE, torch_dtype=dtype, attn_implementation="eager")
    disable_causal_mask(llm)
    llm = PeftModel.from_pretrained(llm, model_dir)
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
    # Default: return all non-zero features; let caller (OS pipeline) prune.
    # top_k in request body caps the number if provided.
    k = int(data.get("top_k", 0))  # 0 => no cap
    if not texts:
        return {"response": [], "features": []}
    enc = ctx["tok"](texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt").to(ctx["device"])
    with torch.no_grad():
        u = ctx["splare"](enc["input_ids"], enc["attention_mask"])
    out = []
    for row in u:
        if k > 0:
            vals, idx = torch.topk(row, k)
            items = zip(idx.tolist(), vals.tolist())
        else:
            # Return all non-zero features
            nz = torch.nonzero(row, as_tuple=False).squeeze(-1)
            vals = row[nz]
            items = zip(nz.tolist(), vals.tolist())
        pairs = {}
        for i, v in items:
            if v > 0:
                pairs[f"f_{i:05d}"] = round(float(v), 4)
        out.append(pairs)
    return {"response": out, "features": out}


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
