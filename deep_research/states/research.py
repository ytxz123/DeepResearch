#***********************************************
#      Filename: state_research.py
#   Description: 研究智能体结构化字段定义
#***********************************************

"""Research Agent的State 字段定义
本文件定义了用于Reasearch Agent工作流程的State对象和结构化字段
"""

from typing_extensions import TypedDict, Annotated, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

# ===== STATE DEFINITIONS =====

class ResearcherState(TypedDict):
    """Research Agent的State，包含消息历史记录和元数据。
    此状态跟踪Researcher的对话，以及正在研究的研究主题和压缩后的研究结果。
    """
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_topic: str
    compressed_research: str

class ResearcherOutputState(TypedDict):
    """Research Agent的输出状态，包含最终的研究结果。"""
    compressed_research: str
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]


# ===== STRUCTURED OUTPUT SCHEMAS =====

class Summary(BaseModel):
    """用于网页内容摘要的结构化字段"""
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")
