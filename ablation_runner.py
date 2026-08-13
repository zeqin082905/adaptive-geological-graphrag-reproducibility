"""
ablation_runner.py — 消融实验驱动脚本
════════════════════════════════════════════════════════════════════════════════
管理以下4个消融变体的运行：

  变体1：完整系统         (full)          — 双路检索 + RASCEF + 子问题分解
  变体2：w/o 双路检索     (no_dual)       — 只用文本向量路，关闭图谱路
  变体3：w/o RASCEF       (no_rascef)     — 用原版 MASTER_PROMPT，关闭RASCEF
  变体4：w/o 子问题分解   (no_decomp)     — 关闭反馈触发，直接用初步检索
  变体5：Naive RAG 基线   (naive_rag)     — 只用向量检索（已有baseline_naive_rag.py）

用法：
    # 运行全部变体（需要时间）
    python ablation_runner.py --all --test-file test_ablation_40.json

    # 只运行某个变体
    python ablation_runner.py --variant no_rascef --test-file test_ablation_40.json

    # 汇总已有结果（不重新运行）
    python ablation_runner.py --summarize

依赖：需要在本仓库根目录下运行
注意：本脚本生成的是 Windows 端运行指令，在 Windows 上执行
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# 消融变体配置
# ══════════════════════════════════════════════════════════════════════════════
ABLATION_VARIANTS = {
    "full": {
        "label": "完整系统",
        "description": "双路混合检索 + RASCEF结构化提示词 + 反馈驱动子问题分解",
        "env_vars": {
            "GEO_ABLATION_DUAL_RETRIEVAL": "1",   # 启用双路检索
            "GEO_ABLATION_RASCEF": "1",            # 启用RASCEF
            "GEO_ABLATION_DECOMP": "1",            # 启用子问题分解
        },
        "output": "results_ablation_full.json",
    },
    "no_dual": {
        "label": "w/o 双路检索",
        "description": "去掉图谱路，只用文本向量检索",
        "env_vars": {
            "GEO_ABLATION_DUAL_RETRIEVAL": "0",   # 关闭双路检索（只用文本路）
            "GEO_ABLATION_RASCEF": "1",
            "GEO_ABLATION_DECOMP": "1",
        },
        "output": "results_ablation_no_dual.json",
        "note": "需要在 retriever.py 中添加环境变量判断逻辑",
    },
    "no_rascef": {
        "label": "w/o RASCEF",
        "description": "使用原版 MASTER_PROMPT，不使用RASCEF结构化框架",
        "env_vars": {
            "GEO_ABLATION_DUAL_RETRIEVAL": "1",
            "GEO_ABLATION_RASCEF": "0",            # 关闭RASCEF，用原版prompt
            "GEO_ABLATION_DECOMP": "1",
        },
        "output": "results_ablation_no_rascef.json",
    },
    "no_decomp": {
        "label": "w/o 子问题分解",
        "description": "关闭反馈驱动子问题分解，直接用初步检索结果",
        "env_vars": {
            "GEO_ABLATION_DUAL_RETRIEVAL": "1",
            "GEO_ABLATION_RASCEF": "1",
            "GEO_ABLATION_DECOMP": "0",            # 关闭子问题分解
        },
        "output": "results_ablation_no_decomp.json",
    },
    "naive_rag": {
        "label": "Naive RAG 基线",
        "description": "纯文本向量检索，无图谱，无RASCEF，无分解",
        "env_vars": {},
        "output": "results_ablation_naive_rag.json",
        "use_baseline_script": "baseline_naive_rag.py",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 生成运行指令
# ══════════════════════════════════════════════════════════════════════════════
def generate_run_commands(test_file: str, project_dir: str = ".") -> str:
    """
    生成 Windows PowerShell 运行指令。
    由于 eval_compare.py 需要在 Windows 项目目录运行，本脚本生成指令供复制执行。
    """
    lines = [
        "# ═══════════════════════════════════════════════════",
        "# geo_graphrag 消融实验运行指令（PowerShell）",
        "# 在本仓库根目录下执行以下命令",
        "# ═══════════════════════════════════════════════════",
        "",
        f"cd {project_dir}",
        "",
    ]

    for variant_key, config in ABLATION_VARIANTS.items():
        lines.append(f"# ── 变体: {config['label']} ──────────────────")
        lines.append(f"# {config['description']}")

        if config.get("use_baseline_script"):
            # Naive RAG 用专用脚本
            lines.append(
                f"python {config['use_baseline_script']} "
                f"--test-file {test_file} "
                f"--output {config['output']}"
            )
        else:
            # 通过环境变量控制
            env_str = "; ".join(
                f"$env:{k}='{v}'"
                for k, v in config["env_vars"].items()
            )
            lines.append(env_str)
            lines.append(
                f"python eval_compare.py "
                f"--mode geo_graphrag "
                f"--test-file {test_file} "
                f"--output {config['output']}"
            )
        lines.append("")

    lines += [
        "# ── 评测打分（在 Linux/Mac 上运行，或本机运行）──",
        "# 先把所有 results_ablation_*.json 复制到评测机器",
        "",
        "# 运行 LLM 五维打分",
    ]
    for config in ABLATION_VARIANTS.values():
        scored = config['output'].replace('.json', '_scored.json')
        lines.append(
            f"python eval_llm_score.py --input {config['output']} --output {scored}"
        )

    lines += [
        "",
        "# 生成消融实验对比表（LaTeX）",
        "python eval_llm_score.py --ablation \\",
    ]
    scored_files = [c['output'].replace('.json', '_scored.json')
                    for c in ABLATION_VARIANTS.values()]
    for i, f in enumerate(scored_files):
        sep = " \\" if i < len(scored_files) - 1 else ""
        lines.append(f"    {f}{sep}")
    lines.append("    --output ablation_final_table.json")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# 生成 eval_compare.py 的 ablation patch 说明
# ══════════════════════════════════════════════════════════════════════════════
PATCH_INSTRUCTIONS = """
# ═══════════════════════════════════════════════════════════════════════════════
# eval_compare.py 消融变体支持补丁说明
# ═══════════════════════════════════════════════════════════════════════════════
#
# 在 eval_compare.py 的 run_single_query() 函数中，读取以下环境变量：
#
# import os
#
# USE_DUAL = os.environ.get("GEO_ABLATION_DUAL_RETRIEVAL", "1") == "1"
# USE_RASCEF = os.environ.get("GEO_ABLATION_RASCEF", "1") == "1"
# USE_DECOMP = os.environ.get("GEO_ABLATION_DECOMP", "1") == "1"
#
# 然后：
# 1. USE_DUAL=0 时：retriever 只调用 text_retriever（跳过 graph_retriever）
# 2. USE_RASCEF=0 时：generator 使用原版 AnswerGenerator（不用 RASCEF 版）
#                     USE_RASCEF=1 时：使用 answer_generator_rascef
# 3. USE_DECOMP=0 时：retriever 跳过子问题分解阶段，直接用 initial_query
#
# 以上三个开关独立控制，可以任意组合。
# ═══════════════════════════════════════════════════════════════════════════════
"""


# ══════════════════════════════════════════════════════════════════════════════
# 汇总已有结果
# ══════════════════════════════════════════════════════════════════════════════
def summarize_existing(result_dir: str = "."):
    """汇总当前目录下已有的消融结果文件"""
    print("\n📊 消融实验结果汇总")
    print("="*70)

    found = []
    for variant_key, config in ABLATION_VARIANTS.items():
        fpath = os.path.join(result_dir, config['output'])
        scored_path = fpath.replace('.json', '_scored.json')

        if os.path.exists(scored_path):
            with open(scored_path, encoding='utf-8') as f:
                data = json.load(f)
            summary = data.get('summary', {})
            found.append((config['label'], summary, scored_path))
            print(f"✅ {config['label']:<20} → {scored_path}")
        elif os.path.exists(fpath):
            print(f"⚠️  {config['label']:<20} → 有结果但未打分: {fpath}")
        else:
            print(f"❌ {config['label']:<20} → 未找到结果文件")

    if len(found) >= 2:
        print("\n── 快速对比（关键词命中率 + LLM均分）──")
        print(f"{'变体':<22} {'命中率':>8} {'LLM均分':>8} {'RAGAS语义':>10}")
        print("-"*52)
        for label, summary, _ in found:
            print(
                f"{label:<22} "
                f"{summary.get('keyword_hit_rate', 0):>8.4f} "
                f"{summary.get('llm_avg', 0):>8.2f} "
                f"{summary.get('ragas_semantic_sim', 0):>10.4f}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 生成论文用的消融实验结果表（模拟数据，用于写作参考）
# ══════════════════════════════════════════════════════════════════════════════
def generate_mock_table():
    """
    生成模拟消融实验结果表，用于论文写作结构参考。
    实际数字在跑完实验后替换。
    """
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    消融实验结果表（模板，数字待填入）                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 系统变体          │ 命中率  │ LLM均分 │ 忠实度  │ 语义相似 │ 上下文精度 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ 完整系统          │  ??.??  │  ?.??   │  0.??   │  0.??   │   0.??    ║
║ w/o 双路检索      │  ??.??  │  ?.??   │  0.??   │  0.??   │   0.??    ║
║ w/o RASCEF        │  ??.??  │  ?.??   │  0.??   │  0.??   │   0.??    ║
║ w/o 子问题分解    │  ??.??  │  ?.??   │  0.??   │  0.??   │   0.??    ║
║ Naive RAG 基线    │  ??.??  │  ?.??   │  0.??   │  0.??   │   0.??    ║
╚══════════════════════════════════════════════════════════════════════════════╝

LaTeX 版本（复制到论文）：

\\begin{table}[h]
\\centering
\\caption{消融实验结果}
\\label{tab:ablation}
\\begin{tabular}{lccccc}
\\hline
\\textbf{系统变体} & \\textbf{命中率} & \\textbf{LLM均分} & \\textbf{忠实度} & \\textbf{语义相似} & \\textbf{上下文精度} \\\\
\\hline
完整系统          & \\textbf{??.??} & \\textbf{?.??} & \\textbf{0.??} & \\textbf{0.??} & \\textbf{0.??} \\\\
w/o 双路检索      & ??.?? & ?.?? & 0.?? & 0.?? & 0.?? \\\\
w/o RASCEF        & ??.?? & ?.?? & 0.?? & 0.?? & 0.?? \\\\
w/o 子问题分解    & ??.?? & ?.?? & 0.?? & 0.?? & 0.?? \\\\
Naive RAG 基线    & ??.?? & ?.?? & 0.?? & 0.?? & 0.?? \\\\
\\hline
\\end{tabular}
\\end{table}
""")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="消融实验驱动脚本")
    parser.add_argument('--all', action='store_true', help='生成全部变体运行指令')
    parser.add_argument('--variant', choices=list(ABLATION_VARIANTS.keys()),
                        help='生成单个变体运行指令')
    parser.add_argument('--test-file', default='test_ablation_40.json',
                        help='消融实验测试集路径')
    parser.add_argument('--summarize', action='store_true',
                        help='汇总已有结果')
    parser.add_argument('--mock-table', action='store_true',
                        help='生成论文用模板表格')
    parser.add_argument('--patch-instructions', action='store_true',
                        help='显示 eval_compare.py 补丁说明')
    args = parser.parse_args()

    if args.summarize:
        summarize_existing()
        return

    if args.mock_table:
        generate_mock_table()
        return

    if args.patch_instructions:
        print(PATCH_INSTRUCTIONS)
        return

    if args.all or args.variant:
        cmds = generate_run_commands(args.test_file)
        print(cmds)
        # 同时保存到文件
        out_file = "ablation_run_commands.ps1"
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write(cmds)
        print(f"\n✅ 运行指令已保存至: {out_file}")
        print("\n" + PATCH_INSTRUCTIONS)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
