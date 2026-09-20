#***********************************************
#      Filename: state_supervisor.py
#   Description: Supervisor智能体的结构化字段定义
#***********************************************

"""
多智能体Supervisor的State定义
本文件定义了多智能体Supervisor工作流程中使用的State对象和tools字段定义。
"""

import operator
from typing_extensions import Annotated, TypedDict, Sequence, List

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from deep_research.states.critique import Critique


class SupervisorState(TypedDict):
    """
    多智能体Supervisor的State定义。
    负责协调Supervisor和Research Agents之间的工作，跟踪研究进展并汇总来自多个Sub-Agent的研究成果。
    """

    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages] # Supervisor信息,用于协调和传递信息
    research_brief: str                                                 # 指导整体研究方向的详细研究简报
    notes: Annotated[list[str], operator.add] = []                      # 已处理和结构化的笔记，可用于生成最终报告
    supervisor_rounds: int = 0                                          # 主管决策轮数，上限由深度档位决定
    critique_nums: int = 0                                              # 跟踪红队批评次数的计数器
    draft_report: str                                                   # 报告草稿
    active_critiques: List[Critique]                                    # 尚未被精修消化的红队批评
    needs_quality_repair: bool                                          # 评估员可以设置一个bool标志，向supervisor发出报告草稿质量低的信号

@tool
class ConductResearch(BaseModel):
    """用于将研究任务委派给专业子Agent (specialized sub-agent) 的工具。"""
    research_topic: str = Field(
        description="研究主题。每次委派的任务应该为单一主题，并需详细描述（至少一个段落）。",
    )

@tool
class ResearchComplete(BaseModel):
    """用于指示研究过程已完成的工具。"""
    pass
