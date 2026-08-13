"""Fail-fast checks for the public, data-free reproducibility package."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", "__pycache__", ".pytest_cache"}
FORBIDDEN_SUFFIXES = {".parquet", ".lance", ".sqlite", ".db", ".npy", ".npz", ".bin"}
FORBIDDEN_NAMES = {"data", "private_data", "restricted_data", "论文实验"}
SECRET_PATTERNS = {
    "OpenAI-style API key": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    "GitHub classic token": re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    "GitHub fine-grained token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def tracked_candidates() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not any(part in IGNORED_PARTS for part in path.parts)
    ]


def main() -> None:
    errors: list[str] = []
    files = tracked_candidates()

    for path in files:
        relative = path.relative_to(ROOT)
        lowered_parts = {part.lower() for part in relative.parts}
        if lowered_parts & FORBIDDEN_NAMES:
            errors.append(f"forbidden data directory: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden data artifact: {relative}")
        if path.stat().st_size > 2_000_000:
            errors.append(f"unexpected file larger than 2 MB: {relative}")

        if path.suffix.lower() in {".py", ".md", ".json", ".txt", ".csv", ".yml", ".yaml", ".cff", ".example"} or path.name in {"LICENSE", ".gitignore"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for label, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"possible {label}: {relative}")

        if path.suffix == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(relative))
            except SyntaxError as exc:
                errors.append(f"Python syntax error in {relative}: {exc}")

    qa_path = ROOT / "examples" / "synthetic_qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    if not isinstance(qa, list) or not qa:
        errors.append("synthetic_qa.json must contain a non-empty list")
    else:
        for index, item in enumerate(qa, start=1):
            missing = {"id", "question", "reference_answer"} - set(item)
            if missing:
                errors.append(f"synthetic QA item {index} is missing: {sorted(missing)}")

    settings_text = (ROOT / "config" / "settings.py").read_text(encoding="utf-8")
    if "sim_threshold: float = 0.75" not in settings_text:
        errors.append("similarity threshold does not match the manuscript (0.75)")
    if "coherence_threshold: float = 0.60" not in settings_text:
        errors.append("coherence threshold does not match the manuscript (0.60)")

    if errors:
        raise SystemExit("Public-release validation failed:\n- " + "\n- ".join(errors))
    print(f"Public-release validation passed for {len(files)} files.")


if __name__ == "__main__":
    main()
