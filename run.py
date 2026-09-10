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
    python run.py                          # 不带参数时进入交互式输入
    python run.py "..." --config config/deepseek.yml   # 切换使用 DeepSeek 配置
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Windows 终端默认非 UTF-8 编码,这里统一改为 UTF-8,
# 避免 argparse --help / 中文提示在控制台出现乱码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.markdown import Markdown

from deep_research import logging as dr_logging
from deep_research.states import AgentInputState
from deep_research.utils import DEFAULT_CONFIG_PATH, resolve_config_path


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # ---- 日志等级 ----
    log_level = args.log_level or os.environ.get("DEEP_RESEARCH_LOG_LEVEL", "INFO")
    dr_logging.setup_logging(log_level)

    # ---- 环境变量覆盖（与 Notebook / README 行为保持一致） ----
    if args.stage:
        os.environ["STAGE"] = args.stage
    if args.config:
        os.environ["CONFIG_PATH"] = args.config

    # 注意：必须在设置 STAGE / CONFIG_PATH 之后再 import agent_builder，
    # 因为各智能体的模型在模块 import 时就会创建（并读取 CONFIG_PATH 选择配置文件）。
    # 若在此前 import，--config 指定的 Key 不会生效，会回退到默认配置文件。
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

    console.print(
        f"[dim]使用配置: {resolve_config_path()} · "
        f"stage: {os.environ.get('STAGE', 'prod')} · 日志级别: {log_level}[/dim]"
    )
    console.print(
        "[dim]调研通常需要 10–20 分钟，期间请保持网络畅通。开始执行……[/dim]"
    )

    # ---- 运行调研 ----
    thread = {"configurable": {"thread_id": "1", "recursion_limit": 50}}
    try:
        # 图中各节点均为 async 函数，必须用异步 API ainvoke 驱动。
        # 脚本环境用 asyncio.run 创建并运行事件循环（nest_asyncio 已兼容 Jupyter）。
        result = asyncio.run(
            full_agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]},
                config=thread,
            )
        )
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
