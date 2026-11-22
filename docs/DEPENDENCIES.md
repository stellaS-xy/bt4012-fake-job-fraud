# Python environments and dependencies

The finished system uses separate environments for classifier inference and
agent/evaluator calls. A machine that only reads reports or stores the model ZIP
needs neither TensorFlow nor an LLM installation. Run commands from the repository
root so paths and repository imports resolve consistently.

## Environment selection

| Task | Requirements | Environment and scope |
|---|---|---|
| Final classifier inference or original final fit | Exported model `runtime_requirements.txt` | Reuse working cluster `.venv-mlp`; exported versions are authoritative |
| Final classifier source dependency reference | [`requirements-final-ensemble.txt`](../requirements-final-ensemble.txt) | Numerical/neural dependencies only; no agent or language-model weights |
| Agent and evaluator | [`agent/requirements-lock.txt`](../agent/requirements-lock.txt) | Separate `agent/.venv`; pinned project snapshot |
| Agent compatible dependency ranges | [`agent/requirements.txt`](../agent/requirements.txt), [`evaluator/requirements.txt`](../evaluator/requirements.txt) | Evaluator requirements include the agent requirements; no classifier environment included |
| Optional Streamlit frontend | [`frontend/requirements.txt`](../frontend/requirements.txt) | Includes agent requirements plus Streamlit |
| Selected offline CI suite | [`tests/requirements-offline.txt`](../tests/requirements-offline.txt) | Small test environment; no TensorFlow, Torch, LangChain, datasets or model weights |
| Earlier exploration and classical experiments | [`requirements.txt`](../requirements.txt) | Broad historical research/UI dependency set; not a lockfile or a complete final-model deployment recipe |
| Notebook 09 MLP cluster experiment | [`requirements-09.txt`](../requirements-09.txt) | Dedicated Linux NVIDIA environment; includes TensorFlow CUDA extra |
| Notebook 10 local diagnostic | [`requirements-10.txt`](../requirements-10.txt) | Optional Torch/Transformers dependencies added to a prepared notebook environment |
| Notebook 10 cluster experiment | [`requirements-10-cluster.txt`](../requirements-10-cluster.txt) | Dedicated environment; see the [DistilBERT guide](../cluster/DISTILBERT_README.md) for the Torch wheel |
| Notebook 13 reference comparison | [`requirements-13.txt`](../requirements-13.txt) | Historical selected experiment environment; Python 3.11.9 in its original instructions |
| Notebook 14 development trials | [`requirements-14.txt`](../requirements-14.txt) | Separate historical development environment; Python 3.11.9 |

Requirements files serve different purposes; do not install all of them into one
environment. The broad root requirements omit the final model's TensorFlow
runtime and cannot replace the exported inference requirements. The final-fit
cluster used Python 3.12.3, TensorFlow 2.21.0 and scikit-learn 1.9.1. The exact
export additionally records NumPy, pandas, SciPy, Keras, joblib and threadpoolctl.
The inference adapter checks all exported versions before loading the weights.
Historical experiment environments can differ from the final export.

## Existing cluster deployment

Reuse `.venv-mlp`, `agent/.venv`, `~/ollama-local` and the existing Ollama model
cache. No reinstall is needed for reading reports or using an intact working
environment. The [deployment guide](DEPLOYMENT.md) explains the model paths and
compute-node checks. Do not import TensorFlow or start Ollama on a login node.

## Fresh agent environment

For a new Linux cluster checkout with a suitable Python 3.12 interpreter:

```bash
python3 -m venv agent/.venv
agent/.venv/bin/python -m pip install -r agent/requirements-lock.txt
```

The lockfile is a recorded dependency snapshot, not a guarantee of compatibility
with every operating system. For a different platform, use the component
requirements and validate that platform separately. The final classifier remains
in its own environment, installed from the trusted export's runtime requirements.
CUDA packages are needed for GPU training, not for the adapter's CPU inference.

## Optional interface

Install `frontend/requirements.txt` only if the interface is required. Starting
Streamlit does not automatically connect a Mac to a remote classifier: the
current adapter starts a local subprocess on the same host as the agent. For a
cluster deployment, keep the agent and classifier on the allocated compute node.
A remote classifier API is not implemented.
