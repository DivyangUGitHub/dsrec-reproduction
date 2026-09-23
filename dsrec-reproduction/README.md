# DSRec Reproduction

A reproducible sequential recommendation system implementing a dual-interest architecture with long-term and short-term sequence modeling, time-aware features, ranking evaluation, and an HTTP inference service.

## Current status

- Dataset preprocessing and sequence construction are implemented.
- DSRec forward pass, training loop, checkpointing, and ranking evaluation are implemented.
- Controlled CPU training has been exercised; full CPU training is computationally expensive.
- This branch adds packaging, inference service scaffolding, containerization, validation, and CI.

> Production deployment still requires an actual CI run, an approved trained model artifact, load testing, observability, authentication/rate limiting as required by the deployment environment, and a completed reproduction report.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
$env:PYTHONPATH="."
python scripts/validate_project.py
pytest
```

## Training

```powershell
$env:PYTHONPATH="."
python scripts/train.py --epochs 1 --max-train-batches 1000 --max-val-batches 20
```

For the complete run, omit the batch limits. CPU execution can be very slow because the current reference implementation uses a Python-level SSM computation.

## Evaluation

```powershell
$env:PYTHONPATH="."
python scripts/evaluate.py --checkpoint data/checkpoints/best.pt
```

## API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Health: `GET /v1/health`

Readiness: `GET /v1/ready`

Recommendations: `POST /v1/recommend`

Example body:

```json
{"user_id": 1, "top_k": 10}
```

## Docker

```bash
docker compose up --build
```

## Reproduction artifacts

Large generated datasets and model checkpoints should be treated as release artifacts rather than ordinary source files. Record the dataset version, preprocessing configuration, seed, model configuration, checkpoint hash, and evaluation metrics in the reproduction report before claiming a final reproduction.
