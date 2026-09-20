#***********************************************
#      Filename: supervisor_agent.py
#   Description: 监督智能体
#***********************************************

"""监督者模式：主管拆解问题并并行委派子研究代理，汇总压缩后的结果用于成稿。

每个子主题拥有独立上下文窗口，互不干扰。
"""


import asyncio
from typing_extensions import Literal
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research.llm import get_chat_model
from deep_research.prompts import CRITICAL_ADDRESS_PROMPT, MULTI_STEP_DENOISE_PROMPT
from deep_research.agents.research_agent import researcher_agent
from deep_research.agents.red_team_agent import red_team_node
from deep_research.agents.evaluator_agent import evaluate_draft_quality
from deep_research.states import (
    SupervisorState,
    ConductResearch,
    ResearchComplete
)
from deep_research.utils import get_today_str, resolve_depth
from deep_research.tools import _think_tool, _refine_draft_report_tool
from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)


# nest_asyncio：兼容 Jupyter 中已存在的事件循环
try:
    import nest_asyncio
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not Jupyter
except ImportError:
    pass


def get_research_notes(messages: list[BaseMessage]) -> list[str]:
    """提取子代理回传的研究笔记。

    只取 ConductResearch 的结果：think_tool 的反思与精修的质量评分属于过程信息，
    混进 findings 会挤占成稿的上下文。
    """
    return [
        msg.content
        for msg in filter_messages(messages, include_types="tool")
        if msg.name == "ConductResearch"
    ]


def get_pending_research_notes(messages: list[BaseMessage]) -> list[str]:
    """提取尚未并入草稿的研究笔记。

    每次精修都会把当时的研究结果写进草稿，所以只取最近一次精修之后的部分，
    避免每轮把全部历史发现重复喂给精修模型。
    """
    pending: list[str] = []
    for msg in filter_messages(messages, include_types="tool"):
        if msg.name == "refine_draft_report":
            pending = []
        elif msg.name == "ConductResearch":
            pending.append(msg.content)
    return pending


# ===== CONFIGURATION =====

supervisor_tools = [ConductResearch, ResearchComplete, _think_tool, _refine_draft_report_tool]
supervisor_model = get_chat_model("supervisor")
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)


# 研究循环规模由深度档位决定（--depth quick|standard|deep）
_depth = resolve_depth()
max_researcher_iterations = _depth["max_iterations"]   # 检索轮数上限（派发过 ConductResearch 的轮次）
max_concurrent_researchers = _depth["max_concurrent"]   # 单轮最大并行子代理数
min_need_repair_score = 6.0                             # 草稿均分低于该值时提醒主管修复

# 决策轮数兜底上限：正常收敛靠 max_researcher_iterations，这一条只拦跑飞的情况。
# 一个检索轮通常还要配一次精修，所以上限按每轮 3 个 cycle 留余量。
max_supervisor_cycles = max_researcher_iterations * 3 + 3


# ===== SUPERVISOR NODES =====

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """决策本轮研究哪些主题、是否并行委派，以及研究是否收尾。"""

    supervisor_messages = state.get("supervisor_messages", [])
    cycle = state.get("supervisor_cycles", 0)
    logger.info(
        "[SUPERVISOR] supervisor invoked (cycle=%d, research_rounds=%d, messages=%d)",
        cycle,
        state.get("research_iterations", 0),
        len(supervisor_messages),
    )

    system_message = MULTI_STEP_DENOISE_PROMPT.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    # 未处理的对抗性反馈注入本轮决策，形成自我纠正；
    # 精修产出新草稿后这些批评即被清空，不会在后续轮次里反复重放。
    pending_critiques = state.get("active_critiques", [])
    if pending_critiques:
        critique_text = "\n".join([f"- {c.author} says: {c.concern}" for c in pending_critiques])
        intervention = SystemMessage(content=CRITICAL_ADDRESS_PROMPT.format(critique_text=critique_text))
        messages.append(intervention)

    # 如果上一次迭代中质量得分较低，则会发出提醒
    if state.get("needs_quality_repair"):
        messages.append(SystemMessage(
            content=f"上一稿报告质量较低（得分低于{min_need_repair_score:.0f}/10），请继续完善。"
        ))

    response = await supervisor_model_with_tools.ainvoke(messages)
    logger.info(
        "supervisor model produced tool_calls=%s num_tool_calls=%d",
        bool(response.tool_calls),
        len(response.tool_calls or []),
    )

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "supervisor_cycles": cycle + 1,
            "needs_quality_repair": False # 在向supervisor发出提醒后，重置修复标志
        }
    )

async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """执行主管本轮的决策：跑反思、并行派发子研究、精修草稿，或结束研究循环。"""

    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # 检索额度用尽后不再派发新检索，但本轮的精修照常执行，
    # 否则最后一轮检索的结果永远进不了草稿
    research_budget_exhausted = research_iterations >= max_researcher_iterations
    # 决策轮兜底只拦跑飞的情况，不参与正常收敛
    cycle_budget_exhausted = state.get("supervisor_cycles", 0) >= max_supervisor_cycles

    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # 满足任一退出条件即结束研究循环
    if cycle_budget_exhausted or no_tool_calls or research_complete:
        # 子代理的压缩结果以 ToolMessage 形式留存，汇总后即为研究笔记
        final_notes = get_research_notes(state.get("supervisor_messages", []))
        logger.info("[REPORT] The research is complete, writing the final report.")

        return Command(
                goto=END,
                update={
                    "notes": final_notes,
                    "research_brief": state.get("research_brief", "")
        })

    else:
        tool_messages = []
        updates = {}
        next_step = "supervisor"

        # 执行所有的工具调用
        try:
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            refine_report_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "refine_draft_report"
            ]

            # 检索额度用尽后拒绝新的检索派发，本轮的 think/refine 照常执行。
            # 每个 tool_call_id 都必须有 ToolMessage 响应，被拒的也要回一条。
            if conduct_research_calls and research_budget_exhausted:
                logger.info(
                    "[SUPERVISOR] research budget exhausted (%d rounds); refusing %d new research calls",
                    research_iterations,
                    len(conduct_research_calls),
                )
                for tool_call in conduct_research_calls:
                    tool_messages.append(
                        ToolMessage(
                            content=(
                                f"检索额度已用尽（已用 {research_iterations} 轮）。不要再派发新的研究任务，"
                                "请把已有发现用 refine_draft_report 并入草稿，然后调用 ResearchComplete。"
                            ),
                            # 不用 ConductResearch 作 name：get_research_notes 按名字收集研究笔记，
                            # 这条提示不是研究发现
                            name="research_skipped",
                            tool_call_id=tool_call["id"]
                        )
                    )
                conduct_research_calls = []

            logger.info(
                "[SUPERVISOR] supervisor_tools executing think=%d conduct=%d refine=%d",
                len(think_tool_calls),
                len(conduct_research_calls),
                len(refine_report_calls),
            )

            # think_tool 先于其他工具执行：其反思结论会影响本轮委派
            for tool_call in think_tool_calls:
                observation = _think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # 并行调用 ConductResearch，每个主题启动一个独立子代理
            if conduct_research_calls:
                # 并行启动多个 research agents
                coros = [
                    researcher_agent.ainvoke({
                        "researcher_messages": [
                            HumanMessage(content=tool_call["args"]["research_topic"])
                        ],
                        "research_topic": tool_call["args"]["research_topic"]
                    })
                    for tool_call in conduct_research_calls
                ]

                # 单个子代理失败不应拖垮其余子代理，失败的那路回一条说明性结果
                tool_results = await asyncio.gather(*coros, return_exceptions=True)

                research_tool_messages = []
                for tool_call, result in zip(conduct_research_calls, tool_results):
                    if isinstance(result, BaseException):
                        logger.error(
                            "[SUPERVISOR] sub-agent failed on topic=%r: %s",
                            tool_call["args"]["research_topic"],
                            result,
                        )
                        content = f"Research failed for this topic: {result}"
                        # 失败说明同样不能进 findings，改用不会与研究笔记混淆的 name
                        name = "research_failed"
                    else:
                        content = result.get("compressed_research", "Error synthesizing research report")
                        name = tool_call["name"]
                    research_tool_messages.append(
                        ToolMessage(
                            content=content,
                            name=name,
                            tool_call_id=tool_call["id"]
                        )
                    )

                tool_messages.extend(research_tool_messages)

            # 用新发现精修草稿并评估质量。
            # 一轮内的多次 refine 入参完全相同（参数由框架注入，模型无法改变），只执行一次、
            # 其余 tool_call 复用结果；每个 tool_call_id 都必须有 ToolMessage 响应，不能丢弃。
            if refine_report_calls:
                if len(refine_report_calls) > 1:
                    logger.info(
                        "[SUPERVISOR] collapsed %d duplicate refine calls into a single execution",
                        len(refine_report_calls),
                    )

                # 只喂还没并入草稿的研究结果：更早的已经写进草稿了
                findings = "\n".join(
                    get_pending_research_notes(list(supervisor_messages) + tool_messages)
                )

                new_draft = await _refine_draft_report_tool.ainvoke({
                    "research_brief": state.get("research_brief", ""),
                    "findings": findings,
                    "draft_report": state.get("draft_report", "")
                })

                eval_result = await evaluate_draft_quality(
                        research_brief=state.get("research_brief", ""),
                        draft_report=new_draft
                )
                logger.info(
                    "[EVALUATOR] comprehensive score=%f, accuracy score=%f, coherence score=%f",
                    eval_result.comprehensiveness_score,
                    eval_result.accuracy_score,
                    eval_result.coherence_score
                )
                logger.info(f"[EVALUATOR] scoing reason: {eval_result.reason}")

                avg_score = (eval_result.comprehensiveness_score + eval_result.accuracy_score + eval_result.coherence_score) / 3

                # 把质量得分追加到tool message, 供Supervisor Agent参考
                tool_messages.extend(
                    ToolMessage(
                        content=f"Draft Updated.\nQuality Score: {avg_score}/10.\nJudge Feedback: {eval_result.reason}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                    for tool_call in refine_report_calls
                )

                updates["draft_report"] = new_draft
                # 新草稿已产出，旧批评视为已消化，不再注入后续轮次
                updates["active_critiques"] = []

                # 低于质量阈值则置位，下一轮主管会收到修复提醒
                if avg_score < min_need_repair_score:
                    updates["needs_quality_repair"] = True

                # 转入 Red Team 对抗审查
                next_step = "red_team"

            # 只有真正派发过检索的轮次才计入检索轮数
            if conduct_research_calls:
                updates["research_iterations"] = research_iterations + 1

            updates["supervisor_messages"] = tool_messages

            return Command(goto=next_step, update=updates)

        except Exception:
            # 该分支会结束整轮研究，必须留日志，否则失败会表现为「报告残缺但无报错」
            logger.exception("[SUPERVISOR] supervisor_tools failed, ending research early")
            return Command(
                goto=END,
                update={
                    "notes": get_research_notes(supervisor_messages),
                    "research_brief": state.get("research_brief", "")
                }
            )



# ===== GRAPH CONSTRUCTION =====

supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("red_team", red_team_node)

supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_edge("supervisor", "supervisor_tools")
supervisor_builder.add_edge("red_team", "supervisor")

supervisor_agent = supervisor_builder.compile()


if __name__ == "__main__":
    print(supervisor_agent.get_graph().draw_ascii())
