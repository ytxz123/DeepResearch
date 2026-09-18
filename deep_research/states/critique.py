#***********************************************
#      Filename: critique.py
#   Description: 批评Agent的格式化输出  
#***********************************************

from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field


class Critique(BaseModel):
    """用于接收来自"Red Team " 或其他质量控制Agent的对抗性反馈的结构化模型"""

    author: str      # 提出批评的 Agent，如 "Red Team"
    concern: str     # 草稿中发现的逻辑漏洞、偏见或事实错误
    addressed: bool = Field(default=False, description="Has the supervisor fixed this?")  # 是否已修复
