from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Generate a compact reproduction report")
    p.add_argument("--stats", default="data/processed/statistics.json")
    p.add_argument("--output", default="docs/reproduction_report.md")
    args = p.parse_args()
    stats_path = Path(args.stats)
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    lines = [
        "# DSRec Reproduction Report",
        "",
        "## Dataset",
        f"- Users: {stats.get('n_users', 'n/a')}",
        f"- Items: {stats.get('n_items', 'n/a')}",
        f"- Interactions: {stats.get('n_interactions', 'n/a')}",
        "",
        "## Validation",
        "- Project validation: run `python scripts/validate_project.py`.",
        "- Unit/integration tests: run `python -m pytest -q`.",
        "- Training/evaluation metrics must be recorded from the actual run; this script does not invent results.",
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
