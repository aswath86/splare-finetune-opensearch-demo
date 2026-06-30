"""Submit v7 training job for model averaging experiment.

Same recipe as v3 (r=32, layer 18, bidirectional, 89k, max_len=192, no warmup).
Only seed varies across submissions. Run 3 times with SEED=1,2,3 to get
three checkpoints for adapter averaging.

Usage:
  SEED=1 HF_TOKEN=... python3 submit_v7.py
  SEED=2 HF_TOKEN=... python3 submit_v7.py
  SEED=3 HF_TOKEN=... python3 submit_v7.py

Each run is ~17h on ml.g5.2xlarge (~$26). With 1× quota, run sequentially
over ~2.5 days, OR request quota bump to 3× for parallel.
"""
import os
from pathlib import Path
from sagemaker.pytorch import PyTorch

ROLE = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::<AWS_ACCOUNT_ID>:role/SplarePocSageMakerRole")
BUCKET = os.environ.get("SPLARE_BUCKET", "splare-poc-<AWS_ACCOUNT_ID>-us-east-1")
DATA_PREFIX = os.environ.get("DATA_PREFIX", "data-89k")
SEED = int(os.environ["SEED"])  # required: 1, 2, or 3

HF_TOKEN = os.environ.get("HF_TOKEN")
assert HF_TOKEN, "HF_TOKEN must be set"

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
        # v3 recipe exactly — only seed varies across runs
        "epochs": 1, "batch-size": 1, "grad-accum": 16,
        "max-len": 192, "n-negs": 8, "lora-rank": 32, "lr": 5e-5,
        "sae-layer": 18, "sae-l0": 116,
        "bidirectional": 1,
        "seed": SEED,
    },
    environment={
        "HF_TOKEN": HF_TOKEN,
        "HF_HOME": "/tmp/hf_cache",
        "TRANSFORMERS_CACHE": "/tmp/hf_cache",
    },
    output_path=f"s3://{BUCKET}/output/",
    base_job_name=f"splare-v7-s{SEED}",
)

print(f"Submitting v7 seed={SEED}")
print(f"Data: s3://{BUCKET}/{DATA_PREFIX}/")
estimator.fit({"train": f"s3://{BUCKET}/{DATA_PREFIX}/"}, wait=False)
j = estimator.latest_training_job.name
print(f"Job: {j}")
print(f"Console: https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/{j}")
