#!/usr/bin/env python
#***********************************************
#      Filename: run.py
#   Description: 深度调研 Agent 的 CLI 启动入口
#   Usage:       python run.py "你的调研问题" [--output xxx.md] [--stage prod] [--config config/qwen.yml]
#***********************************************


"""Deep Research Agent 命令行入口。

用法示例：
    python run.py "帮我写一份关于英伟达最新 GPU 的调研报告"
    python run.py "2026 年多模态大模型的进展如何？" --output report.md
    python run.py "..." --depth quick                  # 快速档，轮数与并行度更小
    python run.py "..." --config config/qwen.yml       # 切换使用 Qwen 配置
    python run.py                                      # 不带参数时进入交互式输入
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Windows 终端默认非 UTF-8，统一改为 UTF-8，避免中文提示在控制台乱码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown

from deep_research import logging as dr_logging
from deep_research.states import AgentInputState
from deep_research.utils import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DEPTH,
    DEPTH_PRESETS,
    resolve_config_path,
)


# 默认输出文件名：output_report_YYYY-MM-DD.md
def default_output_path() -> str:
    return f"results/output_report_{datetime.now().strftime('%Y-%m-%d')}.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="启动 Deep Research 多智能体深度调研 Agent。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            '  python run.py "帮我写一份关于英伟达最新 GPU 的调研报告"\n'
            '  python run.py "..." --output report.md --config config/deepseek.yml\n'
        ),
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="调研问题。省略时进入交互式输入（Ctrl+C 退出）。",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=f"报告保存路径（默认 {default_output_path()}）。",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help='使用的 stage 名称（默认 "prod"，即 config 文件中 stages.prod）。',
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"配置文件路径（默认 {DEFAULT_CONFIG_PATH}，或环境变量 CONFIG_PATH）。",
    )
    parser.add_argument(
        "--depth",
        choices=sorted(DEPTH_PRESETS),
        default=None,
        help="调研深度档位，同时控制研究轮数与并行子代理数（默认 standard，或环境变量 RESEARCH_DEPTH）。",
    )
    parser.add_argument(
        "--no-clarify",
        action="store_true",
        help="关闭开跑前的追问：默认在需求含糊时会先问你一句再开始调研。",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="日志等级，如 DEBUG / INFO / WARNING（默认取环境变量 DEEP_RESEARCH_LOG_LEVEL，缺省 INFO）。",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="调研完成后在终端打印报告全文（默认只打印头部摘要）。",
    )
    return parser.parse_args(argv)


def resolve_question(question: str | None) -> str:
    """问题缺省时进入交互式输入。"""
    if question:
        return question.strip()

    console = Console()
    console.print("[bold cyan]请输入你的调研问题[/bold cyan]（按 [bold]Ctrl+C[/bold] 退出）:")
    try:
        line = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        console.print("\n已取消，未启动调研。")
        raise SystemExit(0)
    if not line:
        console.print("[bold red]问题不能为空。[/bold red]")
        raise SystemExit(1)
    return line


def _ask_user(console: Console, payload: object) -> str:
    """展示 Agent 的追问并读取用户回答；无输入时视为「没有补充」。"""

    question = payload.get("question", "") if isinstance(payload, dict) else str(payload)
    answer = ""
    if question:
        console.print("\n[bold yellow]开始前需要确认一下：[/bold yellow]")
        console.print(question)
        try:
            answer = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print()

    if not answer:
        answer = "没有更多补充，请按现有信息开始调研。"
        console.print(f"[dim]{answer}[/dim]")
    return answer


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- 日志等级 ----
    log_level = args.log_level or os.environ.get("DEEP_RESEARCH_LOG_LEVEL", "INFO")
    dr_logging.setup_logging(log_level)

    # ---- 环境变量覆盖 ----
    if args.stage:
        os.environ["STAGE"] = args.stage
    if args.config:
        os.environ["CONFIG_PATH"] = args.config
    if args.depth:
        os.environ["RESEARCH_DEPTH"] = args.depth

    # agent_builder 在 import 时就会创建各智能体的模型并读取研究档位，
    # 因此上面的环境变量必须先设置好再 import。
    from deep_research.agent_builder import deep_researcher_builder

    console = Console()

    # ---- 解析调研问题 ----
    question = resolve_question(args.question)
    console.rule(f"[bold cyan]Deep Research Agent[/bold cyan] · 调研问题")
    console.print(question)
    console.rule()

    # ---- 编译图（使用内存检查点） ----
    try:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
        full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)
    except Exception as exc:  # 编译失败时给出可读错误
        console.print(f"[bold red]图编译失败:[/bold red] {exc}")
        console.print("请检查 config 文件中的 LLM 配置（base_url / api_key / handle）。")
        return 1

    depth = os.environ.get("RESEARCH_DEPTH") or DEFAULT_DEPTH
    console.print(
        f"[dim]使用配置: {resolve_config_path()} · "
        f"stage: {os.environ.get('STAGE', 'prod')} · 深度: {depth} · 日志级别: {log_level}[/dim]"
    )
    console.print(
        "[dim]调研通常需要 10–20 分钟，期间请保持网络畅通。开始执行……[/dim]"
    )

    # ---- 运行调研（支持中途追问）----
    # recursion_limit 需覆盖主管子图的全部步数：每轮迭代最多 3 步
    # （supervisor + supervisor_tools + red_team），加上追问与首尾节点仍需余量。
    thread = {
        "configurable": {
            "thread_id": "1",
            "recursion_limit": 120,
            "clarify": not args.no_clarify,
        }
    }
    payload: object = {"messages": [{"role": "user", "content": question}]}
    try:
        # 图中节点均为 async 函数，必须用 ainvoke 驱动；
        # 每次 resume 都新起一个事件循环，状态由内存检查点延续。
        while True:
            result = asyncio.run(full_agent.ainvoke(payload, config=thread))
            interrupts = result.get("__interrupt__")
            if not interrupts:
                break
            payload = Command(resume=_ask_user(console, interrupts[0].value))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]已中断，调研未完成。[/bold yellow]")
        return 130
    except Exception as exc:
        console.print(f"\n[bold red]调研执行失败:[/bold red] {exc}")
        return 1

    # ---- 输出报告 ----
    final_report = result.get("final_report", "")
    if not final_report:
        console.print("[bold yellow]未生成最终报告（final_report 为空）。[/bold yellow]")
        return 1

    # 保存文件
    output_path = args.output or default_output_path()
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(final_report)
    except OSError as exc:
        console.print(f"[bold red]报告保存失败:[/bold red] {exc}")
        return 1

    console.rule(f"[bold green]调研完成[/bold green] · 报告已保存到 {output_path}")

    if args.print_report:
        console.print(Markdown(final_report))
    else:
        # 只打印报告开头，提示完整内容见文件
        head = final_report.strip().splitlines()[:20]
        console.print(Markdown("\n".join(head)))
        console.print("[dim]……（完整报告见输出文件）[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
