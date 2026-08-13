"""
geo_graphrag 批量评估脚本
用法: python batch_eval.py [--test-file test_cases.json] [--output report.json]
"""
import json
import subprocess
import sys
import time
import argparse
import re
from datetime import datetime
from pathlib import Path


def run_query(question: str, project_dir: str, timeout: int = 300) -> dict:
    """调用 main.py 执行单条问题，返回原始输出和系统答案"""
    start = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "query", question],
            cwd=project_dir,
            capture_output=True,
            timeout=timeout,
        )
        elapsed = time.time() - start
        # Windows 控制台输出可能是 GBK/CP936，自动检测
        def decode_bytes(b):
            for enc in ("utf-8", "gbk", "gb2312", "cp936"):
                try:
                    return b.decode(enc)
                except Exception:
                    pass
            return b.decode("utf-8", errors="replace")
        stdout = decode_bytes(result.stdout) + decode_bytes(result.stderr)

        # 提取【系统答案】块
        answer_match = re.search(
            r"={10,}\s*【系统答案】\s*={10,}(.*?)(?:={10,}|$)",
            stdout,
            re.DOTALL,
        )
        answer_text = answer_match.group(1).strip() if answer_match else stdout.strip()

        return {
            "success": result.returncode == 0,
            "answer": answer_text,
            "raw_output": stdout,
            "elapsed": round(elapsed, 1),
            "error": None,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "answer": "",
            "raw_output": "",
            "elapsed": timeout,
            "error": "TIMEOUT",
        }
    except Exception as e:
        return {
            "success": False,
            "answer": "",
            "raw_output": "",
            "elapsed": time.time() - start,
            "error": str(e),
        }


def normalize_text(s: str) -> str:
    """
    归一化文本，消除常见的写法差异：
    1. 全角数字圆圈：○〇⊙ → 0
    2. 中文数字简化：一〇三 → 103（用于大队编号）
    3. 简称扩展：地矿局 ↔ 地质矿产勘查开发局
    4. 统一小写
    """
    # 圆圈数字统一
    s = s.replace("○", "0").replace("〇", "0").replace("⊙", "0")
    # 中文数字大队号（一〇三→103，一〇二→102 等）
    cn_digit_map = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
                    "六": "6", "七": "7", "八": "8", "九": "9", "零": "0", "〇": "0", "○": "0"}
    # 处理"一〇三""一〇二"这类三位数队号
    import re
    def replace_cn_num(m):
        return "".join(cn_digit_map.get(c, c) for c in m.group())
    s = re.sub(r"[一二三四五六七八九零〇○][〇○零一二三四五六七八九]{1,2}", replace_cn_num, s)
    # 简称统一：地矿 ↔ 地质矿产勘查
    s = s.replace("地质矿产勘查开发局", "地矿局").replace("地质矿产勘查局", "地矿局")
    s = s.replace("地质矿勘查开发局", "地矿局")
    # 统一小写
    return s.lower()


def evaluate_answer(answer: str, expected_keywords: list) -> dict:
    """
    评估答案质量：
    - 精确匹配：关键词直接出现在答案中
    - 归一化匹配：处理○/〇/简称等写法差异后匹配
    - 全部命中 → CORRECT，部分命中 → PARTIAL，全部未命中 → WRONG
    """
    answer_norm = normalize_text(answer)
    hits = []
    misses = []
    match_types = {}  # 记录匹配方式

    for kw in expected_keywords:
        # 1. 精确匹配
        if kw.lower() in answer.lower():
            hits.append(kw)
            match_types[kw] = "exact"
        # 2. 归一化匹配
        elif normalize_text(kw) in answer_norm:
            hits.append(kw)
            match_types[kw] = "normalized"
        else:
            misses.append(kw)

    hit_rate = len(hits) / len(expected_keywords) if expected_keywords else 0

    if hit_rate == 1.0:
        verdict = "CORRECT"
    elif hit_rate > 0:
        verdict = "PARTIAL"
    else:
        verdict = "WRONG"

    return {
        "verdict": verdict,
        "hit_rate": round(hit_rate, 2),
        "hits": hits,
        "misses": misses,
        "match_types": match_types,
    }


def run_batch(test_file: str, project_dir: str, output_file: str):
    with open(test_file, encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"\n{'='*60}")
    print(f"  geo_graphrag 批量评估")
    print(f"  测试题数量: {len(test_cases)}")
    print(f"  项目目录: {project_dir}")
    print(f"{'='*60}\n")

    results = []
    stats = {"CORRECT": 0, "PARTIAL": 0, "WRONG": 0, "ERROR": 0}

    for tc in test_cases:
        qid = tc["id"]
        question = tc["question"]
        expected = tc["expected_keywords"]
        pred_type = tc.get("predicate", "")

        print(f"[{qid:02d}/{len(test_cases)}] [{pred_type}] {question}")
        print(f"       期望: {expected}")

        run_result = run_query(question, project_dir)

        if not run_result["success"] or run_result["error"]:
            verdict_info = {
                "verdict": "ERROR",
                "hit_rate": 0,
                "hits": [],
                "misses": expected,
            }
            stats["ERROR"] += 1
            print(f"       ❌ ERROR: {run_result['error']} ({run_result['elapsed']}s)\n")
        else:
            verdict_info = evaluate_answer(run_result["answer"], expected)
            stats[verdict_info["verdict"]] += 1
            icon = {"CORRECT": "✅", "PARTIAL": "⚠️", "WRONG": "❌"}.get(
                verdict_info["verdict"], "?"
            )
            print(
                f"       {icon} {verdict_info['verdict']} "
                f"(命中率:{verdict_info['hit_rate']:.0%}, 耗时:{run_result['elapsed']}s)"
            )
            if verdict_info["misses"]:
                print(f"       未命中: {verdict_info['misses']}")
        print()

        results.append(
            {
                **tc,
                "run": run_result,
                "eval": verdict_info,
            }
        )

    # 汇总
    total = len(test_cases)
    correct_rate = stats["CORRECT"] / total
    partial_rate = (stats["CORRECT"] + stats["PARTIAL"]) / total

    print(f"{'='*60}")
    print(f"  评估完成")
    print(f"  ✅ 完全正确: {stats['CORRECT']}/{total} ({correct_rate:.0%})")
    print(f"  ⚠️  部分正确: {stats['PARTIAL']}/{total}")
    print(f"  ❌ 完全错误: {stats['WRONG']}/{total}")
    print(f"  💥 运行错误: {stats['ERROR']}/{total}")
    print(f"  综合通过率(含部分): {partial_rate:.0%}")
    print(f"{'='*60}\n")

    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": total,
            "correct": stats["CORRECT"],
            "partial": stats["PARTIAL"],
            "wrong": stats["WRONG"],
            "error": stats["ERROR"],
            "correct_rate": round(correct_rate, 3),
            "partial_rate": round(partial_rate, 3),
        },
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已保存: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="geo_graphrag 批量评估")
    parser.add_argument(
        "--test-file", default="test_cases.json", help="测试题文件路径"
    )
    parser.add_argument(
        "--project-dir",
        default=str(Path(__file__).resolve().parent),
        help="geo_graphrag 项目根目录",
    )
    parser.add_argument(
        "--output", default="eval_report.json", help="评估报告输出路径"
    )
    args = parser.parse_args()
    run_batch(args.test_file, args.project_dir, args.output)
