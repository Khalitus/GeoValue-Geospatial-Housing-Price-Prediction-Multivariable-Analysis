"""Create a complete, private GeoValue archive INCLUDING local CSV, JSON, and trained models.

This is the safe way to save a true snapshot of your own Mac project.
No files from your working directory are deleted or modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".vercel", "node_modules", ".idea", ".vscode"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}


def create_archive(root: Path, destination: Path) -> tuple[int, int]:
    root = root.resolve()
    dest = destination.resolve()
    files = []
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        rel = candidate.relative_to(root)
        if (any(part in EXCLUDED_PARTS or part.startswith("._") for part in rel.parts)
                or candidate.suffix in EXCLUDED_SUFFIXES or rel.name == ".DS_Store"
                or candidate.resolve() == dest):
            continue
        files.append(candidate)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(dest, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for candidate in sorted(files):
            archive.write(candidate, arcname=f"GeoValue/{candidate.relative_to(root).as_posix()}")
    return len(files), dest.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "GeoValue-Task4-Full-With-Local-Data.zip")
    args = parser.parse_args()
    number, size = create_archive(ROOT, args.output)
    print(f"Created {args.output}: {number} files, {size / (1024**2):.2f} MiB")
    print("Includes existing CSV, JSON, model files, and geographical outputs if present in this project.")


if __name__ == "__main__":
    main()
