#***********************************************
#      Filename: agent_builder.py
#   Description: 多智能体深度研究Builder 
#***********************************************


from typing import Literal

from langchain_core.messages import HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from deep_research.utils import get_today_str
from deep_research.states import AgentState, AgentInputState, ClarifyDecision
from deep_research.prompts import FINAL_REPORT_PROMPT, QUERY_CLARIFY_PROMPT
from deep_research.agents import write_research_brief, write_draft_report
from deep_research.agents import supervisor_agent
from deep_research.llm import get_chat_model, with_structured_output
from deep_research import logging as dr_logging


logger = dr_logging.get_logger(__name__)


# ===== Config =====

writer_model = get_chat_model("writer")
clarify_model = get_chat_model("draft")

# 追问次数上限：超过后直接用已有信息开始调研，避免反复打断
max_clarify_rounds = 2


# ===== CLARIFICATION =====

async def clarify(state: AgentState, config: RunnableConfig) -> dict:
    """判断用户需求是否明确；信息不足时先追问，而不是直接开跑。

    追问依赖检查点（interrupt），只有调用方在 config 中显式开启
    configurable.clarify 时才生效，其余场景（Notebook、API 调用）直接跳过。
    """

    if not config.get("configurable", {}).get("clarify", False):
        return {"clarify_question": ""}

    if state.get("clarify_rounds", 0) >= max_clarify_rounds:
        return {"clarify_question": ""}

    prompt = QUERY_CLARIFY_PROMPT.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str(),
    )
    structured_model = with_structured_output(clarify_model, ClarifyDecision, "draft")
    decision = structured_model.invoke([HumanMessage(content=prompt)])

    if not decision.need_clarification:
        return {"clarify_question": ""}

    return {
        "clarify_question": decision.question,
        "clarify_rounds": state.get("clarify_rounds", 0) + 1,
    }


async def ask_clarify(state: AgentState) -> dict:
    """抛出追问并挂起，等待用户回答；回答作为新消息回到对话后再判断一次。"""

    answer = interrupt({"question": state.get("clarify_question", "")})
    return {"messages": [HumanMessage(content=str(answer))]}


def route_clarify(state: AgentState) -> Literal["ask_clarify", "write_research_brief"]:
    """有待确认的问题就先追问，否则进入研究简报。"""

    return "ask_clarify" if state.get("clarify_question") else "write_research_brief"


# ===== FINAL REPORT GENERATION =====

async def final_report_generation(state: AgentState):
    """最终报告的生成: 用户query，研究简报，findings, 报告初稿 => 报告
    """

    # 取出所有的notes
    notes = state.get("notes", [])
    findings = "\n".join(notes)

    # 组装prompt
    final_report_prompt = FINAL_REPORT_PROMPT.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str(),
        draft_report=state.get("draft_report", "")
    )

    # 生成最后的报告
    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])
    report_text = (final_report.content or "").strip()

    if not report_text:
        # 成稿偶发返回空内容。草稿此时已经过多轮精修，直接退回草稿，
        # 避免整轮调研因为最后一步而全部作废；同时记录可诊断的元信息。
        logger.warning(
            "final report generation returned empty content "
            "(finish_reason=%s, refusal=%s); falling back to the draft",
            final_report.response_metadata.get("finish_reason"),
            final_report.additional_kwargs.get("refusal"),
        )
        report_text = state.get("draft_report", "")

    return {
        "final_report": report_text,
        "messages": ["最终的报告: " + report_text],
    }


# ===== 图的构建 =====

# 构建深度研究的workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# 添加节点
deep_researcher_builder.add_node("clarify", clarify)
deep_researcher_builder.add_node("ask_clarify", ask_clarify)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("write_draft_report", write_draft_report)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# 添加边（追问可多轮，回答后回到 clarify 再次判断）
deep_researcher_builder.add_edge(START, "clarify")
deep_researcher_builder.add_conditional_edges(
    "clarify",
    route_clarify,
    {
        "ask_clarify": "ask_clarify",
        "write_research_brief": "write_research_brief",
    },
)
deep_researcher_builder.add_edge("ask_clarify", "clarify")
deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
deep_researcher_builder.add_edge("write_draft_report", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# 编译graph
agent = deep_researcher_builder.compile()
