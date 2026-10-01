# DSRec Reproduction

A reproducible sequential recommendation system implementing a dual-interest architecture with long-term and short-term sequence modeling, time-aware features, ranking evaluation, and an HTTP inference service.

## Current status

- Dataset preprocessing, training, checkpointing, evaluation, FastAPI inference, PostgreSQL interaction tracking, and the Next.js frontend are implemented.
- Inference converts model-internal item IDs back to original MovieLens item IDs before returning recommendations.
- Like/skip/open feedback is persisted for future offline model improvement; it is not applied to the live model in real time.

## Local setup

From the dsrec-reproduction/ directory:

1. Create and activate a Python virtual environment.
2. Run: pip install -r requirements-dev.txt
3. Run: python scripts/download_movielens.py
4. Run: python scripts/preprocess.py
5. Run: python scripts/build_time_buckets.py
6. Run: python scripts/validate_project.py
7. Run: pytest

The generated processed artifacts are required by inference. The trained checkpoint is a generated binary artifact and is intentionally not committed to source control.

## API

Run: uvicorn src.api.app:app --host 0.0.0.0 --port 8000

Endpoints: GET /v1/health, GET /v1/ready, POST /v1/recommend, POST /v1/interactions, and GET /v1/users/{user_id}/interactions?limit=50.

Readiness checks both the recommender and database connectivity.

## Frontend

From frontend/: npm install && npm run dev

Open http://localhost:3000. Next.js proxies /v1/* to FastAPI. Docker Compose sets DSREC_API_INTERNAL_URL so the web container reaches the API container.

## Docker Compose

Run: docker compose up --build

Open http://localhost:3000 for the UI and http://localhost:8000/docs for the API.

The API mounts data/processed and data/checkpoints read-only. A fresh clone must therefore obtain/generate the processed artifacts and data/checkpoints/best.pt before recommendation inference can be served.

## Reproduction artifacts

Large generated datasets and checkpoints are deliberately excluded from normal Git history. Record dataset/version, preprocessing configuration, seed, model configuration, checkpoint hash, and evaluation metrics for each reproduction run.

## Production hardening

Before public deployment, add authentication/rate limiting, structured observability, backups, and versioned database migrations. The current stack is a reproducible research/demo service.

## Research-paper alignment notes

- `configs/paper_movielens.yaml` captures the paper's reported MovieLens settings: D=64, SSM state=32, convolution width=4, expansion=2, dropout=0.2, maximum sequence length=200, training batch=2048, validation batch=4096, and Adam learning rate=0.001. The paper does not specify MovieLens block count, time-bucket count, or epoch count; those values are documented assumptions in `docs/paper_audit.md`.
- The model includes separate long- and short-interest representations, historical-mean aggregation, log-scaled quantile time buckets, time-gated short-term state updates, detached residual cross-fusion, a tied item-embedding prediction matrix, and full-softmax cross-entropy.
- `HR@K`, `NDCG@K`, and `MRR@K` are reported. Evaluation defaults to the last-item test target and masks items in the input context, except the held-out target. Candidate filtering is not specified by the paper, so use `--allow-seen-items` to run without this assumption.
- The official Mamba backend is selected by the paper config and requires the optional `mamba-ssm` package in a supported Linux/CUDA environment. The portable `torch` backend used by the default config is a custom SSM-style approximation, **not an exact replacement for official Mamba**. Do not compare its scores as an exact paper reproduction without stating this difference.
- This repository currently preprocesses MovieLens-1M only. The paper also evaluates Amazon-Beauty and Amazon-Video-Games; those dataset pipelines and full three-dataset experiments remain outstanding.
