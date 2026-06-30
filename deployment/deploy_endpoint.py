"""Deploy SPLARE v3 as a SageMaker real-time endpoint.

Packages the extracted v3 LoRA adapter (artifacts_v3/) together with the
inference code (src/inference.py, src/splare.py, src/load_sae.py), uploads to S3,
and deploys an ml.g5.2xlarge endpoint.

Requires:
  HF_TOKEN           — HuggingFace token with access to google/gemma-2-2b
  SPLARE_BUCKET      — S3 bucket for the tarball (default: splare-poc-<AWS_ACCOUNT_ID>-us-east-1)
  SAGEMAKER_ROLE     — IAM role ARN with SageMakerFullAccess + the bucket
  ENDPOINT_NAME      — endpoint name (default: splare-v3-ep)
"""
import os
import shutil
import tarfile
from pathlib import Path

import boto3
from sagemaker.pytorch import PyTorchModel

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT / "artifacts_v3"
SRC_DIR = ROOT / "src"

BUCKET = os.environ.get("SPLARE_BUCKET", "splare-poc-<AWS_ACCOUNT_ID>-us-east-1")
ROLE = os.environ.get("SAGEMAKER_ROLE", "arn:aws:iam::<AWS_ACCOUNT_ID>:role/SplarePocSageMakerRole")
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "splare-v3-ep")
HF_TOKEN = os.environ["HF_TOKEN"]

WORK = Path("/tmp/splare_endpoint_pkg")
if WORK.exists():
    shutil.rmtree(WORK)
(WORK / "code").mkdir(parents=True)

# Copy adapter + tokenizer + configs to root of the tarball
for f in ARTIFACT_DIR.iterdir():
    if f.name == "model.tar.gz":
        continue
    shutil.copy(f, WORK / f.name)

# Copy inference code under code/ (SageMaker convention)
for name in ("inference.py", "splare.py", "load_sae.py", "requirements.txt"):
    shutil.copy(SRC_DIR / name, WORK / "code" / name)

tar_path = "/tmp/splare_v3_endpoint.tar.gz"
with tarfile.open(tar_path, "w:gz") as t:
    t.add(str(WORK), arcname=".")
print(f"Built {tar_path} ({os.path.getsize(tar_path)/1e6:.1f} MB)")

s3_key = "endpoints/splare_v3_endpoint.tar.gz"
boto3.client("s3").upload_file(tar_path, BUCKET, s3_key)
model_data = f"s3://{BUCKET}/{s3_key}"
print(f"Uploaded {model_data}")

model = PyTorchModel(
    model_data=model_data,
    role=ROLE,
    entry_point="inference.py",
    framework_version="2.3.0",
    py_version="py311",
    env={
        "HF_TOKEN": HF_TOKEN,
        "HF_HOME": "/tmp/hf_cache",
        "TRANSFORMERS_CACHE": "/tmp/hf_cache",
    },
)

model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.2xlarge",
    endpoint_name=ENDPOINT_NAME,
    wait=True,
)
print(f"Endpoint deployed: {ENDPOINT_NAME}")
