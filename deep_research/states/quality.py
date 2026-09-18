#***********************************************
#      Filename: quality.py
#   Description: 质量控制格式化输出  
#***********************************************

from typing_extensions import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field

class QualityMetric(TypedDict):
    """某一轮迭代中草稿质量的快照"""

    score: float      # Evaluator 打出的质量得分（三维均分）
    feedback: str     # 得分理由
    iteration: int    # 所属迭代轮次，用于观察质量随轮次的变化
