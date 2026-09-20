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
    ResearchComplete,
    QualityMetric
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


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """提取子代理回传的研究笔记。

    子代理的压缩结果以 ToolMessage 形式保存在主管消息历史中，汇总后即为最终报告的研究素材。
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]



# ===== CONFIGURATION =====

supervisor_tools = [ConductResearch, ResearchComplete, _think_tool, _refine_draft_report_tool]
supervisor_model = get_chat_model("supervisor")
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)


# 研究循环规模由深度档位决定（--depth quick|standard|deep）
_depth = resolve_depth()
max_researcher_iterations = _depth["max_iterations"]   # think_tool/ConductResearch/refine 的总调用次数上限
max_concurrent_researchers = _depth["max_concurrent"]   # 单轮最大并行子代理数
min_need_repair_score = 6.0                             # 草稿均分低于该值时提醒主管修复


# ===== SUPERVISOR NODES =====

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """决策本轮研究哪些主题、是否并行委派，以及研究是否收尾。"""
    supervisor_messages = state.get("supervisor_messages", [])
    iteration = state.get("research_iterations", 0)
    logger.info("[SUPERVISOR] supervisor invoked (iteration=%d, messages=%d)", iteration, len(supervisor_messages))
 
    system_message = MULTI_STEP_DENOISE_PROMPT.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages
 
    # 未处理的对抗性反馈注入本轮决策，形成自我纠正
    critiques = state.get("active_critiques", [])
    unaddressed = [c for c in critiques if not c.addressed]
    if unaddressed:
        critique_text = "\n".join([f"- {c.author} says: {c.concern}" for c in unaddressed])
        intervention = SystemMessage(content=CRITICAL_ADDRESS_PROMPT.format(critique_text=critique_text))
        messages.append(intervention)

    # 如果上一次迭代中质量得分较低，则会发出提醒
    if state.get("needs_quality_repair"):
        messages.append(SystemMessage(content="上一稿报告质量较低（得分低于7/10），请继续完善。"))

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
            "research_iterations": iteration + 1,
            "needs_quality_repair": False # 在向supervisor发出提醒后，重置修复标志
        }
    )

async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """执行主管本轮的决策：跑反思、并行派发子研究、精修草稿，或结束研究循环。"""
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )

    # 满足任一退出条件即结束研究循环
    if exceeded_iterations or no_tool_calls or research_complete:
        # 子代理的压缩结果以 ToolMessage 形式留存，汇总后即为研究笔记
        final_notes = get_notes_from_tool_calls(state.get("supervisor_messages", []))
        logger.info("[REPORT] The research is complete, writing the final report.")

        return Command(
                goto=END,
                update={
                    "notes": final_notes,
                    "research_brief": state.get("research_brief", "")
        })

    else:
        tool_messages = []
        all_raw_notes = []
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

                tool_results = await asyncio.gather(*coros)

                # 子代理的压缩结果写入 ToolMessage，供 get_notes_from_tool_calls() 汇总
                research_tool_messages = [
                    ToolMessage(
                        content=result.get("compressed_research", "Error synthesizing research report"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for result, tool_call in zip(tool_results, conduct_research_calls)
                ]

                tool_messages.extend(research_tool_messages)

                # 聚合所有的raw notes
                all_raw_notes = [
                    "\n".join(result.get("raw_notes", [])) 
                    for result in tool_results
                ]

            # 用新发现精修草稿并评估质量。
            # 一轮里模型可能发多个 refine 调用（提示词要求「每次 ConductResearch 后
            # 务必 refine」），但 refine_draft_report 的三个入参都是 InjectedToolArg，
            # 模型根本传不了参数，调用之间没有任何差异，重复执行只是把同一道题算
            # N 遍。这里只真正执行一次，其余 tool_call 复用同一结果。
            # 注意不能直接丢掉多余的调用：每个 tool_call_id 都必须有对应的
            # ToolMessage，否则下一轮调用模型时会因「工具调用没有响应」而报错。
            if refine_report_calls:
                if len(refine_report_calls) > 1:
                    logger.info(
                        "[SUPERVISOR] collapsed %d duplicate refine calls into a single execution",
                        len(refine_report_calls),
                    )

                # findings 必须带上本轮 think/research 刚产出的结果：它们此时还在
                # tool_messages 里，尚未写回 state，只读 state 会漏掉这一轮的增量。
                findings = "\n".join(
                    get_notes_from_tool_calls(list(supervisor_messages) + tool_messages)
                )

                new_draft = _refine_draft_report_tool.invoke({
                    "research_brief": state.get("research_brief", ""),
                    "findings": findings,
                    "draft_report": state.get("draft_report", "")
                })

                eval_result = evaluate_draft_quality(
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

                # 低于质量阈值则置位，下一轮主管会收到修复提醒
                updates["quality_history"] = [QualityMetric(
                    score=avg_score,
                    feedback=eval_result.reason,
                    iteration=state.get("research_iterations", 0))
                ]

                if avg_score < min_need_repair_score:
                    updates["needs_quality_repair"] = True

                # 转入 Red Team 对抗审查
                next_step = "red_team"

            updates["supervisor_messages"] = tool_messages
            updates["raw_notes"] = all_raw_notes
            
            return Command(goto=next_step, update=updates)

        except Exception:
            # 该分支会结束整轮研究，必须留日志，否则失败会表现为「报告残缺但无报错」
            logger.exception("[SUPERVISOR] supervisor_tools failed, ending research early")
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
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
