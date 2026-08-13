"""
ablation_eval.py
================
消融实验批量评估脚本：自动跑5组实验，输出对比报告

5组实验（全部使用 RASCEF Prompt，只变换检索模块）：

  G1  Base          : 纯向量检索（禁用 GraphRAG + Reranker + 子问题分解）
  G2  +GraphRAG     : Base + 图谱路（无Reranker、无子问题分解）
  G3  +Reranker     : Base + Reranker（无图谱、无子问题分解）
  G4  +子问题分解    : Base + Reranker + 子问题分解（无图谱）
  G5  Full          : 全部开启（GraphRAG + Reranker + 子问题分解）

用法：
  python ablation_eval.py \
      --test-file   testset_output/testset_all.json \
      --project-dir . \
      --output-dir  ablation_results \
      --groups      G1,G2,G3,G4,G5 \
      --timeout     300

中断恢复：
  再次运行相同命令，已完成的组自动跳过。
  删除对应 ckpt_Gx.json 可单独重跑某组。

调试：
  加 --max-questions 20 只跑前20题验证流程是否正常。
"""

import argparse
import json
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# 5组消融配置（全部使用 RASCEF Prompt，只变换检索模块开关）
# ─────────────────────────────────────────────────────────────────
ABLATION_GROUPS = {
    "G1": {
        "name":          "Base（纯向量）",
        "mode":          "base",
        "desc":          "纯向量检索，无图谱、无Reranker、无子问题分解",
        "flags_summary": "graph=❌  reranker=❌  decompose=❌",
    },
    "G2": {
        "name":          "+GraphRAG",
        "mode":          "no_reranker_no_decompose",
        "desc":          "Base + 图谱路，无Reranker、无子问题分解",
        "flags_summary": "graph=✅  reranker=❌  decompose=❌",
    },
    "G3": {
        "name":          "+Reranker",
        "mode":          "no_graph_no_decompose",
        "desc":          "Base + Reranker，无图谱路、无子问题分解",
        "flags_summary": "graph=❌  reranker=✅  decompose=❌",
    },
    "G4": {
        "name":          "+子问题分解",
        "mode":          "no_graph",
        "desc":          "Base + Reranker + 子问题分解，无图谱路",
        "flags_summary": "graph=❌  reranker=✅  decompose=✅",
    },
    "G5": {
        "name":          "Full（完整系统）",
        "mode":          "full",
        "desc":          "完整系统：GraphRAG + Reranker + 子问题分解",
        "flags_summary": "graph=✅  reranker=✅  decompose=✅",
    },
}


# ─────────────────────────────────────────────────────────────────
# main.py 调用
# ─────────────────────────────────────────────────────────────────

def run_query(question: str, mode: str, project_dir: str, timeout: int) -> dict:
    """调用 main.py query --mode <mode>，返回系统答案和耗时。"""
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "query", question, "--mode", mode],
            cwd=project_dir,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.time() - start

        def decode_bytes(b):
            for enc in ("utf-8", "gbk", "gb2312", "cp936"):
                try:
                    return b.decode(enc)
                except Exception:
                    pass
            return b.decode("utf-8", errors="replace")

        stdout = decode_bytes(result.stdout) + decode_bytes(result.stderr)
        match = re.search(
            r"={10,}\s*【系统答案】\s*={10,}(.*?)(?:={10,}|$)",
            stdout, re.DOTALL,
        )
        answer_text = match.group(1).strip() if match else stdout.strip()
        return {"success": result.returncode == 0, "answer": answer_text,
                "elapsed": round(elapsed, 1), "error": None}
    except subprocess.TimeoutExpired:
        return {"success": False, "answer": "", "elapsed": timeout, "error": "TIMEOUT"}
    except Exception as e:
        return {"success": False, "answer": "", "elapsed": time.time() - start, "error": str(e)}


# ─────────────────────────────────────────────────────────────────
# 打分（关键词匹配 + LLM 兜底）
# ─────────────────────────────────────────────────────────────────

def normalize_text(s: str) -> str:
    s = s.replace("○", "0").replace("〇", "0").replace("⊙", "0")
    cn_map = {"一":"1","二":"2","三":"3","四":"4","五":"5",
              "六":"6","七":"7","八":"8","九":"9","零":"0"}
    s = re.sub(r"[一二三四五六七八九零〇○][〇○零一二三四五六七八九]{1,2}",
               lambda m: "".join(cn_map.get(c, c) for c in m.group()), s)
    s = s.replace("地质矿产勘查开发局", "地矿局").replace("地质矿产勘查局", "地矿局")
    return s.lower()


def keyword_score(answer: str, keywords: list) -> dict:
    if not keywords:
        return {"verdict": "NO_KEYWORDS", "hit_rate": 0, "hits": [], "misses": []}
    anorm = normalize_text(answer)
    hits  = [kw for kw in keywords
             if kw.lower() in answer.lower() or normalize_text(kw) in anorm]
    misses = [kw for kw in keywords if kw not in hits]
    rate   = len(hits) / len(keywords)
    verdict = "CORRECT" if rate == 1.0 else ("PARTIAL" if rate > 0 else "WRONG")
    return {"verdict": verdict, "hit_rate": round(rate, 2), "hits": hits, "misses": misses}


def llm_score(question: str, answer: str, reference: str,
              ollama_url: str, model: str) -> dict:
    if not answer.strip():
        return {"llm_score": 0, "llm_verdict": "WRONG", "reason": "空答案"}
    prompt = (
        f"你是地质问答系统的评测专家。请对系统答案评分。\n\n"
        f"问题：{question}\n参考答案：{reference}\n系统答案：{answer}\n\n"
        f"评分标准：\n2分=包含核心信息且事实正确\n1分=部分正确有遗漏\n0分=错误或无关\n\n"
        f"只输出JSON：{{\"score\":0/1/2,\"reason\":\"一句话\"}}"
    )
    try:
        import requests
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0, "num_predict": 128}},
            timeout=60,
        )
        text = re.sub(r"```(?:json)?\s*|```\s*", "", resp.json().get("response", "")).strip()
        s = text.find("{"); e = text.rfind("}") + 1
        if s != -1 and e > s:
            obj = json.loads(text[s:e])
            sc  = int(obj.get("score", 0))
            return {"llm_score": sc,
                    "llm_verdict": {2:"CORRECT",1:"PARTIAL",0:"WRONG"}.get(sc,"WRONG"),
                    "reason": obj.get("reason", "")}
    except Exception:
        pass
    return {"llm_score": 0, "llm_verdict": "WRONG", "reason": "LLM评分失败"}


def evaluate(question: str, answer: str, reference: str,
             keywords: list, ollama_url: str, model: str) -> dict:
    """关键词先判；WRONG/PARTIAL 时 LLM 兜底，最终以 LLM 为准。"""
    kw = keyword_score(answer, keywords)
    if kw["verdict"] == "CORRECT":
        return {**kw, "llm_score": 2, "llm_verdict": "CORRECT",
                "final_verdict": "CORRECT", "reason": "关键词全中"}
    llm = llm_score(question, answer, reference, ollama_url, model)
    return {**kw, "llm_score": llm["llm_score"], "llm_verdict": llm["llm_verdict"],
            "reason": llm["reason"], "final_verdict": llm["llm_verdict"]}


# ─────────────────────────────────────────────────────────────────
# 单组实验（含断点续存）
# ─────────────────────────────────────────────────────────────────

def run_group(group_id: str, group_cfg: dict, test_cases: list,
              project_dir: str, output_dir: Path,
              timeout: int, ollama_url: str, llm_model: str) -> dict:

    ckpt_path   = output_dir / f"ckpt_{group_id}.json"
    result_path = output_dir / f"result_{group_id}.json"

    if result_path.exists():
        print(f"\n[{group_id}] 已完成，跳过（删除 {result_path.name} 可重跑）")
        with open(result_path, encoding="utf-8") as f:
            return json.load(f)

    if ckpt_path.exists():
        with open(ckpt_path, encoding="utf-8") as f:
            done_results = json.load(f)
        done_ids = {r["id"] for r in done_results}
        print(f"\n[{group_id}] 续存恢复：已完成 {len(done_results)}/{len(test_cases)} 题")
    else:
        done_results, done_ids = [], set()

    mode = group_cfg["mode"]
    print(f"\n{'='*64}")
    print(f"  [{group_id}] {group_cfg['name']}")
    print(f"  {group_cfg['flags_summary']}")
    print(f"  {group_cfg['desc']}")
    print(f"  题目: {len(test_cases)} 题  超时: {timeout}s/题")
    print(f"{'='*64}")

    stats   = defaultdict(int)
    results = list(done_results)

    for tc in [t for t in test_cases if t.get("id") not in done_ids]:
        qid      = tc.get("id", "")
        question = tc.get("question", "")
        ref_ans  = tc.get("answer", tc.get("expected_answer", ""))
        keywords = tc.get("expected_keywords", tc.get("keywords", []))
        qtype    = tc.get("type", "unknown")

        print(f"  [{len(results)+1:03d}/{len(test_cases)}][{group_id}][{qtype}] {question[:48]}")

        run_r = run_query(question, mode, project_dir, timeout)

        if not run_r["success"] or run_r["error"]:
            eval_r = {"final_verdict": "ERROR", "hit_rate": 0,
                      "hits": [], "misses": keywords, "llm_score": 0,
                      "reason": run_r.get("error")}
            print(f"    💥 ERROR: {run_r['error']} ({run_r['elapsed']}s)")
        else:
            eval_r = evaluate(question, run_r["answer"], ref_ans,
                              keywords, ollama_url, llm_model)
            icon = {"CORRECT":"✅","PARTIAL":"⚠️","WRONG":"❌"}.get(eval_r["final_verdict"],"?")
            print(f"    {icon} {eval_r['final_verdict']}  "
                  f"命中:{eval_r.get('hit_rate',0):.0%}  "
                  f"LLM:{eval_r.get('llm_score','?')}分  "
                  f"{run_r['elapsed']}s")

        stats[eval_r["final_verdict"]] += 1
        results.append({
            "id": qid, "type": qtype,
            "question": question, "reference_answer": ref_ans,
            "system_answer": run_r.get("answer", ""),
            "elapsed": run_r.get("elapsed", 0),
            "error": run_r.get("error"),
            "eval": eval_r,
        })

        if len(results) % 10 == 0:
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False)

    total   = len(results)
    correct = stats.get("CORRECT", 0)
    partial = stats.get("PARTIAL", 0)
    wrong   = stats.get("WRONG",   0)
    error   = stats.get("ERROR",   0)

    type_stats: dict = defaultdict(lambda: defaultdict(int))
    for r in results:
        type_stats[r["type"]][r["eval"].get("final_verdict", "ERROR")] += 1

    summary = {
        "group_id": group_id, "group_name": group_cfg["name"],
        "mode": mode, "flags": group_cfg["flags_summary"],
        "total": total, "correct": correct, "partial": partial,
        "wrong": wrong, "error": error,
        "correct_rate": round(correct / total, 3) if total else 0,
        "partial_rate": round((correct + partial) / total, 3) if total else 0,
        "type_stats": {t: dict(v) for t, v in type_stats.items()},
        "timestamp": datetime.now().isoformat(),
    }

    output = {"summary": summary, "results": results}
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    if ckpt_path.exists():
        ckpt_path.unlink()

    print(f"\n  ── {group_id} 汇总 ──────────────────")
    print(f"  ✅ 正确:    {correct}/{total} ({correct/total:.1%})")
    print(f"  ⚠️  部分:    {partial}/{total} ({partial/total:.1%})")
    print(f"  ❌ 错误:    {wrong}/{total}")
    print(f"  💥 异常:    {error}/{total}")
    print(f"  综合通过率: {(correct+partial)/total:.1%}")
    return output


# ─────────────────────────────────────────────────────────────────
# 对比报告（终端 + JSON + Markdown）
# ─────────────────────────────────────────────────────────────────

def print_comparison(all_results: dict, output_dir: Path):
    print(f"\n{'='*72}")
    print("  消融实验对比汇总")
    print(f"{'='*72}")
    print(f"{'组别':<6} {'名称':<16} {'模块配置':<32} {'正确率':>7} {'综合率':>7}")
    print("-" * 72)

    rows = []
    for gid in ["G1","G2","G3","G4","G5"]:
        if gid not in all_results:
            continue
        s = all_results[gid]["summary"]
        print(f"{gid:<6} {s['group_name']:<16} {s['flags']:<32} "
              f"{s['correct_rate']:>6.1%} {s['partial_rate']:>6.1%}")
        rows.append(s)
    print(f"{'='*72}")

    # 按题型细分
    all_types = sorted({t for d in all_results.values()
                        for t in d["summary"].get("type_stats", {})})
    if all_types:
        print(f"\n按题型正确率：")
        gids = [g for g in ["G1","G2","G3","G4","G5"] if g in all_results]
        print(f"  {'题型':<22}" + "".join(f"{g:>9}" for g in gids))
        print("  " + "-" * (22 + 9 * len(gids)))
        for qt in all_types:
            line = f"  {qt:<22}"
            for gid in gids:
                ts = all_results[gid]["summary"].get("type_stats",{}).get(qt,{})
                n  = sum(ts.values())
                c  = ts.get("CORRECT", 0)
                line += f" {c/n:>7.1%} " if n else f"{'—':>8} "
            print(line)

    # JSON
    report_path = output_dir / "ablation_comparison.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "groups": rows,
            "type_breakdown": {g: all_results[g]["summary"].get("type_stats",{})
                               for g in all_results},
        }, f, ensure_ascii=False, indent=2)

    # Markdown
    gids = [g for g in ["G1","G2","G3","G4","G5"] if g in all_results]
    md = [
        "# 消融实验结果",
        f"\n> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "\n## 总体结果\n",
        "| 组别 | 名称 | 模块配置 | 正确率 | 综合率 | 正确 | 部分 | 错误 | 总计 |",
        "|------|------|---------|--------|--------|------|------|------|------|",
    ]
    for s in rows:
        md.append(f"| {s['group_id']} | {s['group_name']} | {s['flags']} | "
                  f"{s['correct_rate']:.1%} | {s['partial_rate']:.1%} | "
                  f"{s['correct']} | {s['partial']} | {s['wrong']} | {s['total']} |")
    if all_types:
        md += ["\n## 按题型正确率\n",
               "| 题型 | " + " | ".join(gids) + " |",
               "|" + "------|" * (1 + len(gids))]
        for qt in all_types:
            cells = []
            for gid in gids:
                ts = all_results[gid]["summary"].get("type_stats",{}).get(qt,{})
                n  = sum(ts.values())
                c  = ts.get("CORRECT", 0)
                cells.append(f"{c/n:.1%}" if n else "—")
            md.append(f"| {qt} | " + " | ".join(cells) + " |")

    md_path = output_dir / "ablation_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n✓ 对比报告已保存：")
    print(f"  {report_path}")
    print(f"  {md_path}")


# ─────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="消融实验批量评估")
    ap.add_argument("--test-file",     required=True)
    ap.add_argument("--project-dir",   required=True)
    ap.add_argument("--output-dir",    default="ablation_results")
    ap.add_argument("--groups",        default="G1,G2,G3,G4,G5")
    ap.add_argument("--timeout",       type=int, default=300)
    ap.add_argument("--ollama-url",    default="http://localhost:11434")
    ap.add_argument("--llm-model",     default="qwen2.5:14b")
    ap.add_argument("--max-questions", type=int, default=None,
                    help="每组最多评测N题（调试用）")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_file, encoding="utf-8") as f:
        raw = json.load(f)
    test_cases = raw.get("questions", raw) if isinstance(raw, dict) else raw
    if args.max_questions:
        test_cases = test_cases[:args.max_questions]

    groups_to_run = [g.strip() for g in args.groups.split(",")]
    unknown = [g for g in groups_to_run if g not in ABLATION_GROUPS]
    if unknown:
        print(f"✗ 未知实验组: {unknown}，可选: {list(ABLATION_GROUPS.keys())}")
        sys.exit(1)

    print(f"{'='*64}")
    print(f"  消融实验批量评估")
    print(f"{'='*64}")
    print(f"  测试集   : {args.test_file} ({len(test_cases)}题)")
    print(f"  项目目录 : {args.project_dir}")
    print(f"  实验组   : {args.groups}")
    print(f"  LLM评分  : {args.llm_model} @ {args.ollama_url}")
    print(f"\n  实验组配置：")
    for gid in groups_to_run:
        g = ABLATION_GROUPS[gid]
        print(f"  {gid}  {g['name']:<16}  {g['flags_summary']}")

    t_start = time.time()
    all_results = {}
    for gid in groups_to_run:
        all_results[gid] = run_group(
            gid, ABLATION_GROUPS[gid], test_cases,
            args.project_dir, output_dir,
            args.timeout, args.ollama_url, args.llm_model,
        )

    print_comparison(all_results, output_dir)
    print(f"\n✓ 全部完成！总耗时 {(time.time()-t_start)/3600:.1f} 小时")


if __name__ == "__main__":
    main()
