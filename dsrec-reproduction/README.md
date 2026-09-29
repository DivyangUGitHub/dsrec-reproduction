# DSRec Reproduction

A reproducible sequential recommendation system implementing a dual-interest architecture with long-term and short-term sequence modeling, time-aware features, ranking evaluation, and an HTTP inference service.

## Current status

- Dataset preprocessing and sequence construction are implemented.
- DSRec forward pass, training loop, checkpointing, and ranking evaluation are implemented.
- Controlled CPU training has been exercised; full CPU training is computationally expensive.
- Packaging, inference service scaffolding, containerization, validation, and CI are implemented.
- Step 13 adds a Next.js + React frontend for the live recommendation experience.
- Step 14 adds PostgreSQL-backed users and interaction events (impression/click/like/dislike/hide).

> Production deployment still requires load testing, observability, authentication/rate limiting, database migration management, backups, and a completed reproduction report.

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

Record user interaction: `POST /v1/interactions`

List a user's recent interactions: `GET /v1/users/{user_id}/interactions?limit=50`

Recommendation example:

```json
{"user_id": 1, "top_k": 10}
```

Interaction example:

```json
{
  "user_id": 1,
  "event_type": "like",
  "item_id": 575,
  "recommendation_rank": 1,
  "recommendation_score": 5.26,
  "session_id": "demo-session",
  "metadata": {"source": "web"}
}
```

## Frontend

The product UI lives in `frontend/` and uses Next.js + React. Run it separately during development:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

The UI can request recommendations and send like/skip/open events back to the FastAPI interaction API.

## Docker

```bash
docker compose up --build
```

The Compose stack now starts PostgreSQL and the DSRec API. PostgreSQL data is persisted in the `dsrec-postgres` named volume.

## Reproduction artifacts

Large generated datasets and model checkpoints should be treated as release artifacts rather than ordinary source files. Record the dataset version, preprocessing configuration, seed, model configuration, checkpoint hash, and evaluation metrics in the reproduction report before claiming a final reproduction.
