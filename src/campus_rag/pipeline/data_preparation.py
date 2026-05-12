"""
校园文档数据准备模块
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .chunking_config import CHUNKING_CONFIG
from .document_ingestion import load_documents as load_campus_documents
from .document_ingestion import normalize_text

# 创建当前模块的日志记录器
logger = logging.getLogger(__name__)


class DataPreparationModule:
    """数据准备模块 - 负责校园文档加载、清洗和分块"""

    CATEGORY_LABELS = ["规章制度", "教务教学", "校园生活", "通知公告", "其他"]

    def __init__(self, data_path: str):
        """
        初始化数据准备模块
        Args:
            data_path: 数据文件夹路径
        """
        # 初始化数据路径
        self.data_path = data_path
        # 初始化文档列表
        self.documents: List[Document] = []
        # 初始化分块列表
        self.chunks: List[Document] = []
        # 初始化父子映射
        self.parent_child_map: Dict[str, str] = {}
        # 初始化父文档映射
        self.parent_documents_map: Dict[str, Document] = {}

    def load_documents(self) -> List[Document]:
        """
        加载完整父文档数据
        Returns:
            加载的完整父文档列表
        """
        logger.info("正在从 %s 加载校园文档...", self.data_path)
        # 文档列表
        documents = load_campus_documents(self.data_path)
        self.documents = documents
        # 初始化派生状态
        self._reset_loaded_state()

        for parent_doc in self.documents:
            self._normalize_parent_metadata(parent_doc)
            parent_id = parent_doc.metadata.get("parent_id")
            if parent_id:
                self.parent_documents_map[parent_id] = parent_doc

        logger.info("成功加载 %s 个父文档", len(self.documents))
        return documents

    def _reset_loaded_state(self) -> None:
        """清空派生状态，避免重复加载时串数据"""
        # 分块结果
        self.chunks = []
        # 子块到父文档的映射
        self.parent_child_map = {}
        # 父文档索引
        self.parent_documents_map = {}

    def _normalize_parent_metadata(self, parent_doc: Document) -> None:
        """补齐校园文档的标准元数据"""
        metadata = parent_doc.metadata or {}
        source = metadata.get("source", "")
        source_path = Path(source) if source else Path(self.data_path)

        doc_title = metadata.get("doc_title") or metadata.get("source_name") or source_path.stem or "未知文档"
        doc_category = metadata.get("doc_category") or "其他"
        department = metadata.get("department") or ""
        file_type = metadata.get("file_type") or source_path.suffix.lstrip(".").lower()
        parent_id = metadata.get("parent_id") or metadata.get("doc_id")
        if not parent_id:
            parent_seed = metadata.get("relative_path") or source_path.as_posix() or doc_title
            parent_id = hashlib.md5(parent_seed.encode("utf-8")).hexdigest()
        doc_id = metadata.get("doc_id") or parent_id

        metadata.update(
            {
                "doc_title": doc_title,
                "doc_category": doc_category,
                "department": department,
                "file_type": file_type,
                "parent_id": parent_id,
                "doc_id": doc_id,
                "source": source,
                "doc_type": "parent",
            }
        )
        # 父文档不保存 section；章节/页码位置由子块 metadata 表达
        metadata.pop("section", None)
        parent_doc.metadata = metadata

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        """对外提供支持的分类标签列表"""
        return list(cls.CATEGORY_LABELS)

    def chunk_documents(self) -> List[Document]:
        """
        校园文档结构感知分块
        Returns:
            分块后的文档块列表
        """
        logger.info("正在进行校园文档分块...")

        if not self.documents:
            raise ValueError("请先加载文档")

        chunks: List[Document] = []
        self.parent_child_map = {}

        for parent_doc in self.documents:
            parent_chunks: List[Document] = self._split_parent_document(parent_doc)
            for chunk_index, raw_chunk in enumerate(parent_chunks):
                cleaned_content = normalize_text(raw_chunk.page_content or "")
                if not cleaned_content:
                    continue

                chunk_doc = self._build_chunk_document(
                    parent_doc=parent_doc,
                    raw_chunk=raw_chunk,
                    cleaned_content=cleaned_content,
                    chunk_index=chunk_index,
                    batch_index=len(chunks),
                )
                chunks.append(chunk_doc)
                self.parent_child_map[chunk_doc.metadata["chunk_id"]] = chunk_doc.metadata["parent_id"]

        self.chunks = chunks
        logger.info("校园文档分块完成，共生成 %s 个chunk", len(chunks))
        return chunks

    def _split_parent_document(self, parent_doc: Document) -> List[Document]:
        """
        按文件类型选择合适的切分策略
        Args:
            parent_doc: 父文档
        Returns:
            List[Document]: 分块后的文档块列表
        """
        file_type = (parent_doc.metadata.get("file_type") or "").lower()

        if file_type == "md":
            return self._split_markdown_document(parent_doc)
        if file_type == "txt":
            return self._split_text_document(parent_doc, CHUNKING_CONFIG["txt"])
        if file_type == "pdf":
            return self._split_text_document(parent_doc, CHUNKING_CONFIG["pdf"])
        # 其他文件类型，返回原始内容
        return [Document(page_content=parent_doc.page_content, metadata={})]

    def _split_markdown_document(self, parent_doc: Document) -> List[Document]:
        """Markdown 文档优先按标题切分，没有标题则退回通用分块"""
        markdown_config = CHUNKING_CONFIG["md"]
        fallback_config = markdown_config["fallback"]
        # 检查是否有标题
        if not self._has_markdown_headers(parent_doc.page_content):
            return self._split_text_document(parent_doc, fallback_config)

        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[tuple(header) for header in markdown_config["headers_to_split_on"]],
            strip_headers=bool(markdown_config["strip_headers"]),
        )

        try:
            markdown_chunks = splitter.split_text(parent_doc.page_content)
        except Exception as exc:
            logger.warning("Markdown 结构分割失败: %s, error=%s", parent_doc.metadata.get("source"), exc)
            markdown_chunks = []

        chunks: List[Document] = []
        for chunk in markdown_chunks:
            if not (chunk.page_content or "").strip():
                continue
            chunks.append(Document(page_content=chunk.page_content, metadata=dict(chunk.metadata or {})))

        if chunks:
            return chunks
        # 没有有效分块，调用通用分块配置
        return self._split_text_document(parent_doc, fallback_config)

    def _split_text_document(
        self,
        parent_doc: Document,
        config: Dict[str, Any],
    ) -> List[Document]:
        """使用递归字符分割器按配置切分普通文本"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(config["chunk_size"]),
            chunk_overlap=int(config["chunk_overlap"]),
            separators=list(config["separators"]),
        )

        raw_chunks = splitter.split_text(parent_doc.page_content)
        chunks: List[Document] = []
        for raw_chunk in raw_chunks:
            if raw_chunk.strip():
                chunks.append(Document(page_content=raw_chunk, metadata={}))

        if chunks:
            return chunks
        # 没有有效分块，返回原始内容
        if (parent_doc.page_content or "").strip():
            return [Document(page_content=parent_doc.page_content, metadata={})]

        return []

    def _build_chunk_document(
        self,
        parent_doc: Document,
        raw_chunk: Document,
        cleaned_content: str,
        chunk_index: int,
        batch_index: int,
    ) -> Document:
        """
        把分块内容转成带标准元数据的子文档
        Args:
            parent_doc: 父文档
            raw_chunk: 原始分块文档
            cleaned_content: 清理后的分块内容
            chunk_index: 分块索引
            batch_index: 批次索引
        Returns:
            Document: 带标准元数据的子文档
        """
        parent_metadata = dict(parent_doc.metadata or {})
        chunk_metadata = dict(raw_chunk.metadata or {})
        parent_id = parent_metadata.get("parent_id") or parent_metadata.get("doc_id")
        if not parent_id:
            parent_seed = parent_metadata.get("relative_path") or parent_metadata.get("source") or cleaned_content[:128]
            parent_id = hashlib.md5(str(parent_seed).encode("utf-8")).hexdigest()

        chunk_id = self._make_chunk_id(parent_id, chunk_index)
        section = self._infer_section(parent_doc, chunk_metadata)

        metadata = {
            **parent_metadata,
            **chunk_metadata,
            "parent_id": parent_id,
            "doc_id": parent_metadata.get("doc_id") or parent_id,
            "doc_type": "child",
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
            "batch_index": batch_index,
            "chunk_size": len(cleaned_content),
            "section": section,
            "doc_title": parent_metadata.get("doc_title", "未知文档"),
            "doc_category": parent_metadata.get("doc_category", "其他"),
            "department": parent_metadata.get("department", ""),
            "file_type": parent_metadata.get("file_type", ""),
            "source": parent_metadata.get("source", ""),
            "relative_path": parent_metadata.get("relative_path", ""),
            "page": parent_metadata.get("page"),
        }

        return Document(page_content=cleaned_content, metadata=metadata)

    def _infer_section(self, parent_doc: Document, chunk_metadata: Dict[str, Any]) -> str:
        """
        推断分块的章节标题
        Args:
            parent_doc: 父文档
            chunk_metadata: 分块元数据
        Returns:
            str: 推断出的分块的章节标题
        """
        section_parts: List[str] = []
        for key in ("h1", "h2", "h3"):
            value = chunk_metadata.get(key)
            if value:
                section_parts.append(str(value).strip())
        if section_parts:
            return " / ".join(section_parts)

        page = parent_doc.metadata.get("page")
        if page is not None:
            return f"第{page}页"

        return "正文"

    @staticmethod
    def _has_markdown_headers(text: str) -> bool:
        """判断文本中是否存在 Markdown 标题"""
        for line in text.splitlines()[:20]:
            if line.lstrip().startswith("#"):
                return True
        return False

    @staticmethod
    def _make_chunk_id(parent_id: str, chunk_index: int) -> str:
        """
        生成稳定的文档块ID，便于向量索引和BM25结果去重
        Args:
            parent_id: 父文档ID
            chunk_index: 分块索引
        Returns:
            str: 稳定的文档块ID
        """
        raw_id = f"{parent_id}:{chunk_index}"
        return hashlib.md5(raw_id.encode("utf-8")).hexdigest()

    def get_context_documents(self, retrieved_chunks: List[Document], window_size: int = 1) -> List[Document]:
        """
        根据命中的子块构建用于生成回答的证据窗口文档
        子块负责召回，生成阶段只回填命中块及其相邻块，避免把完整父文档塞进上下文
        Args:
            retrieved_chunks: 检索到的文档块列表
            window_size: 每个命中块前后回填的相邻块数量
        Returns:
            按父文档聚合后的证据上下文文档列表
        """
        if not retrieved_chunks:
            return []

        try:
            context_window = max(0, int(window_size))
        except (TypeError, ValueError):
            context_window = 1
        # 按 parent_id 和 chunk_index 建立的子块查找表
        chunks_by_parent = self._build_chunks_by_parent()
        # 父文档的召回强度信息
        parent_rank_info: Dict[str, Dict[str, Any]] = {}
        # 父文档的命中块索引
        selected_indices_by_parent: Dict[str, Set[int]] = {}
        # 父文档的兜底块
        fallback_chunks_by_parent: Dict[str, List[Document]] = {}

        for chunk_rank, chunk in enumerate(retrieved_chunks):
            chunk_metadata = chunk.metadata or {}
            # 从 chunk 元数据中获取父文档ID
            parent_id = chunk_metadata.get("parent_id")
            if not parent_id:
                continue
            # 读取 chunk 的 RRF分数，默认值为0.0
            raw_score = chunk_metadata.get("rrf_score")
            try:
                chunk_score = float(raw_score) if raw_score is not None else 0.0
            except (TypeError, ValueError):
                chunk_score = 0.0
            # 汇总每个父文档的命中信息
            if parent_id not in parent_rank_info:
                parent_rank_info[parent_id] = {
                    "first_rank": chunk_rank, # 第一个命中块的排名
                    "best_score": chunk_score, # 最佳命中块的RRF分数
                    "hit_count": 1, # 命中次数
                }
            else:
                rank_info = parent_rank_info[parent_id]
                rank_info["hit_count"] += 1
                rank_info["first_rank"] = min(rank_info["first_rank"], chunk_rank)
                rank_info["best_score"] = max(rank_info["best_score"], chunk_score)
            
            chunk_index = self._safe_chunk_index(chunk_metadata.get("chunk_index"))
            # 如果 chunk_index 无效，将该块添加到兜底块列表
            if chunk_index is None:
                fallback_chunks_by_parent.setdefault(parent_id, []).append(chunk)
                continue
            # 获取父文档的子块查找表
            parent_chunks = chunks_by_parent.setdefault(parent_id, {})
            # 将当前块添加到父文档的子块查找表中
            parent_chunks.setdefault(chunk_index, chunk)
            # 将当前块的索引添加到父文档的命中块索引中
            selected_indices = selected_indices_by_parent.setdefault(parent_id, set())

            # 将当前块的相邻块索引添加到父文档的命中块索引中
            for context_index in range(chunk_index - context_window, chunk_index + context_window + 1):
                if context_index in parent_chunks:
                    selected_indices.add(context_index)

        # 按照第一个命中块的排名、最佳命中块的RRF分数、命中次数对父文档进行排序
        sorted_parent_ids = sorted(
            parent_rank_info.keys(),
            key=lambda parent_id: (
                parent_rank_info[parent_id]["first_rank"],
                -parent_rank_info[parent_id]["best_score"],
                -parent_rank_info[parent_id]["hit_count"],
            ),
        )

        # 构建证据上下文文档
        context_docs: List[Document] = []
        for parent_id in sorted_parent_ids:
            # 获取父文档的子块查找表
            parent_chunks = chunks_by_parent.get(parent_id, {})
            # 获取父文档的命中块索引
            selected_indices = sorted(selected_indices_by_parent.get(parent_id, set()))
            # 从父文档的子块查找表中获取命中块及其相邻块
            context_chunks = [parent_chunks[index] for index in selected_indices if index in parent_chunks]

            if not context_chunks:
                context_chunks = fallback_chunks_by_parent.get(parent_id, [])

            if not context_chunks:
                continue

            page_content = "\n\n".join(
                self._format_context_chunk(chunk)
                for chunk in context_chunks
            )
            metadata = self._build_context_metadata(parent_id, context_chunks, selected_indices, context_window)
            context_docs.append(Document(page_content=page_content, metadata=metadata))

        logger.info(
            "从 %s 个召回块构建 %s 个证据上下文文档，窗口大小=%s",
            len(retrieved_chunks),
            len(context_docs),
            context_window,
        )
        return context_docs

    def _build_chunks_by_parent(self) -> Dict[str, Dict[int, Document]]:
        """
        按 parent_id 和 chunk_index 建立子块查找表
        Args:
            None
        Returns:
            按 parent_id 和 chunk_index 建立的子块查找表，键为 parent_id，值为一个字典，键为 chunk_index，值为对应的证据块
        """
        chunks_by_parent: Dict[str, Dict[int, Document]] = {}
        for chunk in self.chunks:
            metadata = chunk.metadata or {}
            parent_id = metadata.get("parent_id")
            chunk_index = self._safe_chunk_index(metadata.get("chunk_index"))
            if parent_id is None or chunk_index is None:
                continue
            chunks_by_parent.setdefault(parent_id, {})[chunk_index] = chunk
        return chunks_by_parent

    def _build_context_metadata(
        self,
        parent_id: str,
        context_chunks: List[Document],
        selected_indices: List[int],
        context_window: int,
    ) -> Dict[str, Any]:
        """
        构建证据上下文文档的父文档级元数据
        Args:
            parent_id: 父文档 ID
            context_chunks: 包含证据块的列表，每个块包含元数据
            selected_indices: 命中块的索引列表
            context_window: 上下文窗口大小，用于回填相邻块
        Returns:
            元数据字典，包含父文档的元数据和上下文块的元数据
        """
        parent_doc = self._find_parent_document(parent_id)
        metadata = dict(parent_doc.metadata or {}) if parent_doc is not None else {}
        first_chunk_metadata = dict(context_chunks[0].metadata or {})
        # 从第一个命中块的元数据补充缺失的元数据
        for key in ("doc_title", "doc_category", "department", "file_type", "source", "relative_path", "page"):
            if metadata.get(key) is None and first_chunk_metadata.get(key) is not None:
                metadata[key] = first_chunk_metadata[key]

        metadata["parent_id"] = parent_id
        metadata["doc_id"] = metadata.get("doc_id") or first_chunk_metadata.get("doc_id") or parent_id
        metadata["doc_title"] = metadata.get("doc_title", "未知文档")
        metadata["doc_category"] = metadata.get("doc_category", "其他")
        metadata["department"] = metadata.get("department", "")
        metadata["file_type"] = metadata.get("file_type", "")
        metadata["source"] = metadata.get("source", "")
        metadata["doc_type"] = "context"
        metadata["context_window_size"] = context_window
        metadata["context_chunk_indices"] = selected_indices
        metadata.pop("section", None)
        return metadata

    def _find_parent_document(self, parent_id: str) -> Optional[Document]:
        """根据 parent_id 查找完整父文档，仅用于继承元数据"""
        parent_doc = self.parent_documents_map.get(parent_id)
        if parent_doc is not None:
            return parent_doc

        for candidate in self.documents:
            if (candidate.metadata or {}).get("parent_id") == parent_id:
                return candidate

        return None

    @staticmethod
    def _format_context_chunk(chunk: Document) -> str:
        """格式化单个证据块，保留块序号便于定位"""
        metadata = chunk.metadata or {}
        chunk_index = metadata.get("chunk_index")
        label = f"片段 {chunk_index}" if chunk_index is not None else "片段"
        return f"[{label}]\n{chunk.page_content}"

    @staticmethod
    def _safe_chunk_index(value: Any) -> Optional[int]:
        """安全转换 chunk_index"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息
        Returns:
            统计信息字典
        """
        if not self.documents:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "categories": {},
                "departments": {},
                "file_types": {},
                "avg_chunk_size": 0,
            }
        # 分类
        categories: Dict[str, int] = {}
        # 部门
        departments: Dict[str, int] = {}
        # 文件类型
        file_types: Dict[str, int] = {}

        for parent_doc in self.documents:
            metadata = parent_doc.metadata or {}

            category = metadata.get("doc_category", "其他")
            categories[category] = categories.get(category, 0) + 1

            department = metadata.get("department") or "未注明"
            departments[department] = departments.get(department, 0) + 1

            file_type = metadata.get("file_type") or "unknown"
            file_types[file_type] = file_types.get(file_type, 0) + 1
        # 平均块大小
        avg_chunk_size = 0
        if self.chunks:
            avg_chunk_size = sum(chunk.metadata.get("chunk_size", 0) for chunk in self.chunks) / len(self.chunks)

        return {
            "total_documents": len(self.documents), # 总文档数
            "total_chunks": len(self.chunks), # 总块数
            "categories": categories,
            "departments": departments,
            "file_types": file_types,
            "avg_chunk_size": avg_chunk_size,
        }
