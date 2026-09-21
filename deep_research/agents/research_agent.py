#***********************************************
#      Filename: research_agent.py
#   Description:  研究智能体
#***********************************************

"""Research Agent核心实现
该文件实现了一个Research Agent，它可以执行迭代式网络搜索和综合分析，以回答复杂的研究问题。
"""


from typing_extensions import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage

from deep_research.llm import get_chat_model
from deep_research.states import ResearcherState, ResearcherOutputState
from deep_research.utils import get_today_str
from deep_research.tools import _tavily_search_tool, _think_tool
from deep_research.prompts import RESEARCH_AGENT_PROMPT, COMPRESS_RESEARCH_SYSTEM_PROMPT, COMPRESS_RESEARCH_HUMAN_PROMPT
from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)


# ===== CONFIGURATION =====

# 初始化tools
tools = [_tavily_search_tool, _think_tool]
tools_by_name = {tool.name: tool for tool in tools}

# 初始化模型
model = get_chat_model("researcher_main")
model_with_tools = model.bind_tools(tools)
compress_model = get_chat_model("researcher_compressor")

# 检索次数不设代码上限，由提示词里的停止条件自行收敛。
# 唯一的兜底是子图继承的 recursion_limit（run.py 设 120 步），撞上会抛
# GraphRecursionError，该路子代理的结果被 gather 换成一条失败说明。


# ===== AGENT NODES =====

def llm_call(state: ResearcherState):
    """根据当前状态决策下一步的动作"""

    msg_count = len(state.get("researcher_messages", []))
    logger.debug("llm_call invoked with %d messages", msg_count)

    # 组装系统提示词（其中的 {date} 等占位符需在此展开）
    system_message = RESEARCH_AGENT_PROMPT.format(date=get_today_str())

    # 调用大模型
    response = model_with_tools.invoke(
        [SystemMessage(content=system_message)] + state["researcher_messages"]
    )

    logger.info(
        "llm_call produced response tool_calls=%s num_tool_calls=%d",
        bool(response.tool_calls),
        len(response.tool_calls or []),
    )
    return {
        "researcher_messages": [response]
    }

def tool_node(state: ResearcherState):
    """根据前一次大模型结果执行所有工具调用"""

    tool_calls = state["researcher_messages"][-1].tool_calls
    logger.info("tool_node executing %d tool calls", len(tool_calls or []))

    # 单个工具失败（网络波动、限流、后端报错）不该中断整轮研究：
    # 把错误作为观察结果交回模型，让它换查询或转向其他方向。
    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        logger.info("Invoking tool %s with args=%s", tool_call["name"], tool_call["args"])
        try:
            observations.append(tool.invoke(tool_call["args"]))
        except Exception as exc:
            logger.error("Tool %s failed: %s", tool_call["name"], exc)
            observations.append(f"Tool '{tool_call['name']}' failed: {exc}")

    # 获取工具输出
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {
        "researcher_messages": tool_outputs,
    }

def _drop_think_tool(messages: list[BaseMessage]) -> list[BaseMessage]:
    """剔除 think_tool 的调用与结果。

    压缩提示词明确要求忽略 think_tool，把反思发过去只是白付输入费。
    调用与结果必须成对删除：只删 ToolMessage 会留下悬空的 tool_call，接口会报错。
    """

    think_ids = {
        call["id"]
        for msg in messages
        for call in (getattr(msg, "tool_calls", None) or [])
        if call["name"] == "think_tool"
    }
    if not think_ids:
        return list(messages)

    cleaned: list[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id in think_ids:
            continue
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            kept = [call for call in tool_calls if call["id"] not in think_ids]
            if not kept and not msg.content:
                continue  # 只剩空壳的思考消息，留着没有意义
            msg = msg.model_copy(update={"tool_calls": kept})
        cleaned.append(msg)
    return cleaned


def compress_research(state: ResearcherState) -> dict:
    """把研究发现压缩为高价值摘要，只保留有用信息."""

    # 组装prompt
    system_message = COMPRESS_RESEARCH_SYSTEM_PROMPT.format(date=get_today_str())
    history = _drop_think_tool(list(state.get("researcher_messages", [])))
    messages = [SystemMessage(content=system_message)] + history +\
            [HumanMessage(content=COMPRESS_RESEARCH_HUMAN_PROMPT.format(research_topic=state.get("research_topic", "")))]
    logger.info("compress_research invoked with %d messages", len(messages))

    # 调用summary模型
    response = compress_model.invoke(messages)

    return {
        "compressed_research": str(response.content),
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer."""
    messages = state["researcher_messages"]
    last_message = messages[-1]

    decision = "tool_node" if last_message.tool_calls else "compress_research"
    logger.info("should_continue decision=%s (has_tool_calls=%s)", decision, bool(last_message.tool_calls))
    return decision


# ===== GRAPH CONSTRUCTION =====

# Build the agent
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # Continue research loop
        "compress_research": "compress_research", # 返回 final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call") # 继续搜索获得更多结果
agent_builder.add_edge("compress_research", END)

# Compile the agent
researcher_agent = agent_builder.compile()

if __name__ == "__main__":
    print(researcher_agent.get_graph().draw_ascii())
