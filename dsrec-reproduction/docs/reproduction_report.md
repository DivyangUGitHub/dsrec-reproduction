# DSRec Reproduction Report

## 1. Project Overview

This project reproduces the DSRec sequential recommendation pipeline.

The implementation includes:

- Interaction data preprocessing
- User/item mappings
- Time-aware sequence construction
- Training and validation data generation
- DSRec model implementation
- Configurable model/training parameters
- Ablation configurations
- Model checkpoint saving
- Ranking-based evaluation
- Recommendation inference/API components
- Automated tests

---

## 2. Environment

The project was developed and tested in a Python virtual environment.

Main dependencies include:

- Python
- NumPy
- Pandas
- PyTorch
- PyYAML
- FastAPI
- Uvicorn
- Pydantic
- Pytest

Dependencies are specified in:

```text
requirements.txt