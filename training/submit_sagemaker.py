"""Submit SPLARE v3 training job to SageMaker.

Reproduces the winning model:
  - Gemma-2-2B backbone, bidirectional attention
  - Gemma Scope residual SAE at layer 18 (width 65k, L0=116)
  - LoRA r=32, lr=5e-5, 1 epoch, batch 1 × grad_accum 16
  - 89k rows from hanhainebula/bge-multilingual-gemma2-data (MIRACL + Mr.TyDi)

Training takes ~17h on ml.g5.2xlarge (~$26).

Expects s3://$BUCKET/$DATA_PREFIX/train.jsonl — produced by training/prepare_data.py
then uploaded to S3.
"""
import os
from pathlib import Path
from sagemaker.pytorch import PyTorch

ROLE = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::<AWS_ACCOUNT_ID>:role/SplarePocSageMakerRole")
BUCKET = os.environ.get("SPLARE_BUCKET", "splare-poc-<AWS_ACCOUNT_ID>-us-east-1")
DATA_PREFIX = os.environ.get("DATA_PREFIX", "data-89k")

HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "HF_TOKEN must be set"

# src/ is a sibling directory of training/
SRC_DIR = str((Path(__file__).resolve().parent.parent / "src").resolve())

estimator = PyTorch(
    entry_point="train_sagemaker.py",
    source_dir=SRC_DIR,
    role=ROLE,
    instance_type="ml.g5.2xlarge",
    instance_count=1,
    framework_version="2.3.0",
    py_version="py311",
    volume_size=60,
    max_run=22 * 3600,
    hyperparameters={
        "epochs": 1, "batch-size": 1, "grad-accum": 16,
        "max-len": 192, "n-negs": 8, "lora-rank": 32, "lr": 5e-5,
        "sae-layer": 18, "sae-l0": 116,
        "bidirectional": 1,
    },
    environment={
        "HF_TOKEN": HF_TOKEN,
        "HF_HOME": "/tmp/hf_cache",
        "TRANSFORMERS_CACHE": "/tmp/hf_cache",
    },
    output_path=f"s3://{BUCKET}/output/",
    base_job_name="splare-poc",
)

print(f"Data: s3://{BUCKET}/{DATA_PREFIX}/")
print("Submitting...")
estimator.fit({"train": f"s3://{BUCKET}/{DATA_PREFIX}/"}, wait=False)
j = estimator.latest_training_job.name
print(f"Job: {j}")
print(f"Console: https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/{j}")
