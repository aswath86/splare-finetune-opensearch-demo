# SPLARE PoC — OpenSearch-deployable multilingual sparse retriever

SPLARE (Sparse Learned Retrieval with Autoencoders) fine-tuned on Gemma-2-2B +
Gemma Scope SAE, deployed to OpenSearch via a SageMaker remote connector and the
native `sparse_encoding` ingest pipeline.

See [`RESULTS.md`](./RESULTS.md) for the benchmark results.

**Headline**: SPLARE v3 wins **62 / 62** head-to-head evaluation cells against
`opensearch-neural-sparse-encoding-multilingual-v1` (mlv1) across MIRACL, MLDR,
XPQA and MLQA (19 languages, 4 benchmarks). Cross-lingual average advantage:
**+0.40 nDCG@10**.

---

## Project layout

```
sparse-finetune-splare-poc-v2/
├── src/                        # Code bundled into SageMaker training + endpoint jobs
│   ├── splare.py               # SPLARE module (backbone → hidden layer-L → SAE → SPLADE-pool)
│   ├── load_sae.py             # Download Gemma Scope SAE tensors
│   ├── step.py                 # One training step (encode + dot + KL + FLOPS)
│   ├── losses.py               # KL distillation + FLOPS sparsity loss
│   ├── train_sagemaker.py      # Training entry point (v3 recipe: layer 18 + bidir + LoRA r=32)
│   ├── inference.py            # SageMaker endpoint handler (no internal Top-K cap)
│   └── requirements.txt        # Runtime deps for the SageMaker container
│
├── training/                   # Orchestration from the host
│   ├── config.py               # Paper hyperparameters (reference)
│   ├── prepare_data.py         # Build the 89k multilingual training JSONL
│   └── submit_sagemaker.py     # Launch the SageMaker training job
│
├── deployment/
│   ├── deploy_endpoint.py      # Package artifacts_v3 + src/, upload, deploy endpoint
│   └── os_connector_to_sm.py   # Create ml-commons connector + register remote model
│
├── evaluation/                 # Each benchmark is independent
│   ├── _common.py              # Shared: native pipeline, bulk index, neural_sparse, metrics
│   ├── prep_miracl.py          # Build miracl_eval/<lang>/ (10k corpus)
│   ├── prep_mldr.py            # Build mldr_eval/<lang>/  (10k corpus)
│   ├── eval_miracl.py          # MIRACL (fr, ja, de)
│   ├── eval_mldr.py            # MLDR   (fr, de, ja, zh)
│   ├── eval_xpqa.py            # XPQA   (6 configs — downloads on the fly)
│   └── eval_mlqa.py            # MLQA   (49 configs — downloads on the fly, resumable)
│
├── artifacts_v3/               # THE winning LoRA adapter (25 MB) + tokenizer + configs
├── results/                    # Canonical final nDCG@10 numbers per benchmark
│   ├── miracl_scores.json
│   ├── mldr_scores.json
│   ├── xpqa_scores.json
│   └── mlqa_scores.json
│
├── RESULTS.md                  # Benchmark results (nDCG@10 across 4 benchmarks)
├── requirements.txt            # Host-side tooling (datasets, sagemaker, boto3, requests)
└── .gitignore
```

---

## Reproducing the end-to-end path

All AWS steps assume temporary Isengard credentials are exported
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`,
`AWS_DEFAULT_REGION=us-east-1`) and `HF_TOKEN` is set for Gemma access.

### 1. Train (~17h, ~$26 on ml.g5.2xlarge)

```bash
cd training
python prepare_data.py --output data/train.jsonl      # ~89k rows
aws s3 cp data/train.jsonl s3://$SPLARE_BUCKET/data-89k/train.jsonl
HF_TOKEN=hf_... python submit_sagemaker.py
```

Artifacts land at `s3://$SPLARE_BUCKET/output/<job>/output/model.tar.gz`. The
winning v3 model is already checked in under `artifacts_v3/`, so this step is
optional.

### 2. Deploy to SageMaker + wire into OpenSearch

```bash
# From project root
HF_TOKEN=hf_... python deployment/deploy_endpoint.py
# … wait for endpoint InService …
python deployment/os_connector_to_sm.py
# Capture NEW_MODEL_ID from the output.
```

Also register the built-in multilingual-v1 model in OpenSearch (standard
ml-commons flow; not scripted here). Capture its model id.

### 3. Evaluate

```bash
SPLARE_ID=<NEW_MODEL_ID>   # from step 2
MLV1_ID=<mlv1 model id>    # from your OS cluster

# MIRACL (in-distribution): fr, ja, de, 10k corpus each
python evaluation/prep_miracl.py
python evaluation/eval_miracl.py \
  --splare-model-id "$SPLARE_ID" --mlv1-model-id "$MLV1_ID"

# MLDR (OOD Wikipedia long-docs): fr, de, ja, zh, 10k corpus each
python evaluation/prep_mldr.py
python evaluation/eval_mldr.py \
  --splare-model-id "$SPLARE_ID" --mlv1-model-id "$MLV1_ID"

# XPQA (OOD e-commerce QA): 6 configs, corpora downloaded on the fly
python evaluation/eval_xpqa.py \
  --splare-model-id "$SPLARE_ID" --mlv1-model-id "$MLV1_ID"

# MLQA (OOD Wikipedia QA): 49 configs, resumable
python evaluation/eval_mlqa.py \
  --splare-model-id "$SPLARE_ID" --mlv1-model-id "$MLV1_ID"
```

### 4. Tear down

```bash
aws sagemaker delete-endpoint        --endpoint-name splare-v3-ep
aws sagemaker delete-endpoint-config --endpoint-config-name splare-v3-ep
```

---

## Key production detail

The `inference.py` endpoint handler returns **all non-zero sparse features**
(no internal Top-K cap). Pruning is done by OpenSearch at ingest time via the
native `sparse_encoding` processor:

```json
{"sparse_encoding": {
  "model_id": "<splare model id>",
  "field_map": {"text": "text_sparse"},
  "prune_type": "top_k",
  "prune_ratio": 400
}}
```

Doc-side Top-K=400 is asymmetric against query-side natural sparsity
(typically 40–80 features) — this matches the paper's default of `k=40` queries
/ `k=400` docs and is what unlocks the SPLARE-wins-everywhere result. Capping
docs at the endpoint (Top-K=40 for both) gave mlv1 a misleading advantage on
MIRACL in earlier experiments; the asymmetric Top-K (q=40 / docs=400) fixes this.

---

## References

This work is a reproduction and OpenSearch integration of the SPLARE method:

- **Learning Retrieval Models with Sparse Autoencoders** — Thibault Formal, Maxime
  Louis, Hervé Déjean, Stéphane Clinchant (NAVER LABS Europe), ICLR 2026.
  - Abstract: https://arxiv.org/abs/2603.13277
  - PDF: https://arxiv.org/pdf/2603.13277

There is no public SPLARE checkpoint; the model here was built from open components
(Gemma-2-2B + Gemma Scope SAE + a LoRA adapter) following the paper's recipe.
