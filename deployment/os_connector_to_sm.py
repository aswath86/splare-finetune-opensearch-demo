"""Wire OpenSearch → SageMaker endpoint via ml-commons remote connector.

Run after deploy_endpoint.py reports InService. Creates an ml-commons
connector with AWS SigV4 credentials to the SageMaker endpoint, registers
a remote SPARSE_ENCODING model, and deploys it.

Requires:
  AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN  in env
  OPENSEARCH_URL    (default http://localhost:9202)
  ENDPOINT_NAME     (default splare-v3-ep)

Prints NEW_MODEL_ID / NEW_CONNECTOR_ID; capture these to wire pipelines.
"""
import json
import os
import time

import requests

OS_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9202")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
ENDPOINT = os.environ.get("ENDPOINT_NAME", "splare-v3-ep")
MODEL_NAME = os.environ.get("OS_MODEL_NAME", "splare-v3-sm")

AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_TOKEN = os.environ.get("AWS_SESSION_TOKEN", "")

# Undeploy + delete any existing model with the same name
old = requests.post(f"{OS_URL}/_plugins/_ml/models/_search", json={
    "query": {"term": {"name.keyword": MODEL_NAME}}, "size": 10
}).json().get("hits", {}).get("hits", [])
for m in old:
    mid = m["_id"]
    print(f"undeploying/deleting existing model {mid}")
    requests.post(f"{OS_URL}/_plugins/_ml/models/{mid}/_undeploy")
    time.sleep(1)
    requests.delete(f"{OS_URL}/_plugins/_ml/models/{mid}")

# Create SageMaker connector
credential = {"access_key": AWS_KEY, "secret_key": AWS_SECRET}
if AWS_TOKEN:
    credential["session_token"] = AWS_TOKEN

conn = requests.post(f"{OS_URL}/_plugins/_ml/connectors/_create", json={
    "name": f"{MODEL_NAME}-connector",
    "description": "SPLARE v3 on SageMaker",
    "version": "1",
    "protocol": "aws_sigv4",
    "parameters": {"region": REGION, "service_name": "sagemaker"},
    "credential": credential,
    "client_config": {"max_connection": 30, "connection_timeout": 10000, "read_timeout": 120000},
    "actions": [{
        "action_type": "predict",
        "method": "POST",
        "url": f"https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{ENDPOINT}/invocations",
        "headers": {"Content-Type": "application/json"},
        "pre_process_function": "connector.pre_process.default.embedding",
        "request_body": '{"texts": ${parameters.input}}',
    }]
}).json()

if "connector_id" not in conn:
    print("CONNECTOR CREATE FAILED:", json.dumps(conn, indent=2))
    raise SystemExit(1)

cid = conn["connector_id"]
print(f"connector_id={cid}")

reg = requests.post(f"{OS_URL}/_plugins/_ml/models/_register?deploy=true", json={
    "name": MODEL_NAME,
    "function_name": "remote",
    "description": "SPLARE v3 on SageMaker",
    "connector_id": cid,
}).json()
mid = reg["model_id"]
print(f"model_id={mid}")

time.sleep(4)
state = requests.get(f"{OS_URL}/_plugins/_ml/models/{mid}").json().get("model_state")
print(f"state={state}")

# Smoke predict
pred = requests.post(
    f"{OS_URL}/_plugins/_ml/_predict/sparse_encoding/{mid}",
    json={"text_docs": ["test"]},
).json()
print(f"predict result keys: {list(pred.keys())}")
print(f"NEW_MODEL_ID={mid}")
print(f"NEW_CONNECTOR_ID={cid}")
