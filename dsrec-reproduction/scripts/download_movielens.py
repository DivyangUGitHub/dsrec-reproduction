"""
Download and extract the MovieLens-1M dataset (Phase 2).

[PAPER-SPECIFIED] Dataset: MovieLens-1M (Harper & Konstan, 2015), the
target dataset for this reproduction, statistics reported in Table I.

Downloads ml-1m.zip from the official GroupLens mirror, optionally verifies
its MD5 checksum against a value the caller supplies, extracts it, and
places the raw .dat files under data/raw/ml-1m/. Refuses to overwrite an
existing extraction unless --force is passed, so a rerun never silently
clobbers data someone else's script already touched.

NOTE: this repo's sandbox network allowlist does not include
files.grouplens.org, so this script has not been executed end-to-end here
— see the Phase 0 final report for what was and wasn't run.
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
EXPECTED_FILES = ("ratings.dat", "movies.dat", "users.dat")


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` rather than loading the archive into memory."""
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def md5sum(path: Path, chunk_size: int = 1 << 20) -> str:
    """MD5 hex digest of a file, read in chunks so large archives don't blow up memory."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(zip_path: Path, dest_dir: Path) -> Path:
    """Extract ml-1m.zip into dest_dir; verifies the three .dat files landed."""
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    extracted = dest_dir / "ml-1m"
    missing = [f for f in EXPECTED_FILES if not (extracted / f).exists()]
    if missing:
        raise RuntimeError(f"Extraction incomplete, missing: {missing}")
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--url", default=ML1M_URL)
    parser.add_argument(
        "--expected-md5",
        default=None,
        help=(
            "MD5 checksum to verify the archive against. Not hard-coded "
            "here: we could not fetch/confirm the official value from this "
            "sandbox's network [unverified unless you supply it]."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Redo download/extraction if it already exists.")
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    target = args.raw_dir / "ml-1m"
    if target.exists() and not args.force:
        print(f"{target} already exists, skipping. Use --force to redo it.")
        return

    zip_path = args.raw_dir / "ml-1m.zip"
    download(args.url, zip_path)

    digest = md5sum(zip_path)
    print(f"MD5: {digest}")
    if args.expected_md5:
        if digest != args.expected_md5:
            raise RuntimeError(f"Checksum mismatch: got {digest}, expected {args.expected_md5}")
    else:
        print("No --expected-md5 given, skipping checksum verification.")

    extracted = extract(zip_path, args.raw_dir)
    print(f"Extracted to {extracted}")


if __name__ == "__main__":
    main()
