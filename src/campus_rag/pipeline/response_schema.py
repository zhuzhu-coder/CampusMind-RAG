"""
校园知识库响应模式定义
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True) # 数据类，不可变
class RetrievedSource:
    """与答案引用编号关联的召回文档块"""

    source_id: int # 答案引用编号，同一文档的多个文档块可以共享编号
    doc_title: str # 文档标题
    doc_category: str # 文档分类
    department: str # 发布部门
    file_type: str # 文件类型
    section: str # 章节
    source: str # 来源文件路径
    page: Optional[int] # 页码
    chunk_index: int # 文档块索引
    rrf_score: Optional[float] # RRF分数
    snippet: str # 文档块摘要

    def to_dict(self) -> Dict[str, Any]:
        """将 RetrievedSource 对象转换为字典"""
        return asdict(self)


@dataclass(frozen=True)
class RAGTrace:
    """RAG 查询链路调试信息"""

    retrieval_strategy: str # 实际使用的检索策略
    filters: Dict[str, Any] # 显式元数据过滤条件
    timings_ms: Dict[str, float] # 各阶段耗时，单位毫秒
    retrieval_params: Dict[str, Any] # 检索与上下文参数
    retrieved_chunks: List[Dict[str, Any]] # 原始召回块摘要
    context_documents: List[Dict[str, Any]] # 证据窗口文档摘要
    source_count: int # 结构化来源数量

    def to_dict(self) -> Dict[str, Any]:
        """将 RAGTrace 对象转换为字典"""
        return asdict(self)


@dataclass(frozen=True)
class RAGResponse:
    """完整RAG响应"""

    question: str # 原始问题
    route_type: str # 路由类型
    rewritten_query: str # 重写后的问题
    answer: str # 最终答案
    sources: List[RetrievedSource] # 与答案引用编号对齐的召回文档块列表
    trace: Optional[RAGTrace] = None # 可选调试链路

    def to_dict(self) -> Dict[str, Any]:
        """将 RAGResponse 对象转换为字典"""
        payload = {
            "question": self.question,
            "route_type": self.route_type,
            "rewritten_query": self.rewritten_query,
            "answer": self.answer,
            # 将对齐后的召回文档块列表转换为字典列表
            "sources": [source.to_dict() for source in self.sources],
        }
        if self.trace is not None:
            payload["trace"] = self.trace.to_dict()
        return payload
