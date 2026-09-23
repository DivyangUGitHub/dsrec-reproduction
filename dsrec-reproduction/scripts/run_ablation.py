from __future__ import annotations

import argparse
import json
from pathlib import Path


ABLATIONS = {
    "no_cross_fusion": "configs/ablations/no_cross_fusion.yaml",
    "no_dual_interest": "configs/ablations/no_dual_interest.yaml",
    "no_short_ssm": "configs/ablations/no_short_ssm.yaml",
    "dual_mamba": "configs/ablations/dual_mamba.yaml",
}


def main():
    p = argparse.ArgumentParser(description="Prepare an ablation run manifest")
    p.add_argument("--name", choices=sorted(ABLATIONS))
    p.add_argument("--output", default="experiments/results/ablation_manifest.json")
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    payload = {"ablation": args.name, "config": ABLATIONS[args.name], "status": "ready"}
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
