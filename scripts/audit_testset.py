"""Audit existing testsets for leakage, low-difficulty prompts, and module attribution."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def classify(case: dict) -> tuple[str, list[str], str]:
    qtype = case.get("type", "unknown")
    if qtype == "entity_attr":
        return "text_fact", ["text"], "easy"
    if qtype == "entity_rel":
        return "graph_relation", ["graph"], "medium"
    if qtype == "community":
        return "community_reasoning", ["graph", "decompose"], "medium"
    if qtype == "cross_reasoning":
        return "multi_hop_reasoning", ["graph", "decompose", "reranker"], "hard"
    return qtype, [], "unknown"


def leakage_flags(case: dict) -> list[str]:
    flags = []
    question = str(case.get("question", ""))
    answer = str(case.get("answer", ""))
    evidence = str(case.get("evidence", ""))

    if answer and evidence and answer[:24] in evidence:
        flags.append("answer_copyable_from_evidence")

    for field in ("source_entity", "target_entity", "source_community", "center_entity"):
        value = str(case.get(field, "")).strip()
        if value and value in question:
            flags.append(f"question_leaks_{field}")

    overlap = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,20}", question)) & set(
        re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,20}", answer)
    )
    if len(overlap) >= 4:
        flags.append("high_question_answer_overlap")
    return flags


def main():
    parser = argparse.ArgumentParser(description="Audit layered testset quality.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    questions = payload["questions"] if isinstance(payload, dict) and "questions" in payload else payload

    audited = []
    stats = Counter()
    for case in questions:
        layer, modules, difficulty = classify(case)
        flags = leakage_flags(case)
        quality = "keep" if not flags else "review"
        audited_case = {
            **case,
            "layer": layer,
            "required_modules": modules,
            "difficulty": difficulty,
            "audit_flags": flags,
            "quality_label": quality,
        }
        audited.append(audited_case)
        stats[quality] += 1
        for flag in flags:
            stats[flag] += 1

    output = {
        "meta": {
            "version": "audited_v1",
            "total": len(audited),
            "quality_counts": {"keep": stats["keep"], "review": stats["review"]},
            "flag_counts": {k: v for k, v in stats.items() if k not in {"keep", "review"}},
        },
        "questions": audited,
    }
    Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["meta"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
