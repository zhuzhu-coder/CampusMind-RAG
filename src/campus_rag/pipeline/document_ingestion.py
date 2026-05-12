"""
校园文档接入层
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# 支持的文件类型后缀名
SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
# 分类映射表
CATEGORY_MAPPING = {
    "regulations": "规章制度",
    "teaching": "教务教学",
    "life": "校园生活",
    "notices": "通知公告",
}
# 兜底分类
FALLBACK_CATEGORY = "其他"
# 支持的文本编码格式
TEXT_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "utf-16", "utf-16-le", "utf-16-be")


def load_documents(data_path: str | Path) -> List[Document]:
    """
    递归加载校园知识库中的支持文件
    Args:
        data_path: 校园知识库目录路径
    Returns:
        List[Document]: 加载的文档列表
    """
    # 转换为 Path 对象
    root = Path(data_path)
    if not root.exists():
        logger.warning("文档目录不存在: %s", root)
        return []

    documents: List[Document] = []
    # 递归遍历目录树，加载所有支持的文件类型
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # 取文件后缀转小写
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        if suffix == ".md":
            doc = load_markdown_document(path, root)
            if doc is not None:
                documents.append(doc)
        elif suffix == ".txt":
            doc = load_text_document(path, root)
            if doc is not None:
                documents.append(doc)
        elif suffix == ".pdf":
            documents.extend(load_pdf_document(path, root))

    return documents


def load_markdown_document(path: Path, data_root: Optional[Path] = None) -> Optional[Document]:
    """
    读取 Markdown 文档并生成父文档对象
    Args:
        path: Markdown 文档文件路径
        data_root: 校园知识库根目录路径
    Returns:
        Optional[Document]: 生成的文档对象，或 None
    """
    text = _read_text_file(path)
    if not text:
        logger.warning("Markdown 文档为空: %s", path)
        return None
    # 清洗文本
    normalized_text = normalize_text(text)
    if not normalized_text:
        logger.warning("Markdown 文档清洗后为空: %s", path)
        return None

    return Document(
        page_content=normalized_text,
        metadata=build_base_metadata(path, normalized_text, "md", data_root=data_root),
    )


def load_text_document(path: Path, data_root: Optional[Path] = None) -> Optional[Document]:
    """
    读取 TXT 文档并生成父文档对象
    Args:
        path: TXT 文档文件路径
        data_root: 校园知识库根目录路径
    Returns:
        Optional[Document]: 生成的文档对象，或 None
    """
    text = _read_text_file(path)
    if not text:
        logger.warning("TXT 文档为空: %s", path)
        return None

    normalized_text = normalize_text(text)
    if not normalized_text:
        logger.warning("TXT 文档清洗后为空: %s", path)
        return None

    return Document(
        page_content=normalized_text,
        metadata=build_base_metadata(path, normalized_text, "txt", data_root=data_root),
    )


def load_pdf_document(path: Path, data_root: Optional[Path] = None) -> List[Document]:
    """
    按页读取可直接抽取文本的 PDF 文档
    Args:
        path: PDF 文档文件路径
        data_root: 校园知识库根目录路径
    Returns:
        List[Document]: 生成的文档对象列表
    """
    try:
        # 读取 PDF 文档
        loader = PyPDFLoader(str(path))
        pages: List[Document] = loader.load()
    except Exception as exc:  # pragma: no cover - 解析失败属于外部输入问题
        logger.warning("PDF 读取失败: %s, error=%s", path, exc)
        return []

    cleaned_pages: List[Tuple[int, str]] = []
    # 逐页处理
    for page_number, page in enumerate(pages, 1):
        text = normalize_text(page.page_content or "")
        if not text:
            logger.info("PDF 页面无可抽取文本，已跳过: %s, page=%s", path, page_number)
            continue
        # 页码索引
        raw_page_number = page.metadata.get("page")
        try:
            page_index = int(raw_page_number) + 1 if raw_page_number is not None else page_number
        except (TypeError, ValueError):
            page_index = page_number
        cleaned_pages.append((page_index, text))

    if not cleaned_pages:
        logger.warning("PDF 没有可抽取文本，已标记为不支持: %s", path)
        return []

    # PDF 属于同一份文档，标题统一从第一页可抽取文本推断
    pdf_title = infer_doc_title(path, cleaned_pages[0][1], "pdf")
    documents: List[Document] = []
    for page_index, text in cleaned_pages:
        # 构建元数据
        metadata = build_base_metadata(
            path,
            text,
            "pdf",
            page=page_index,
            data_root=data_root,
            doc_title=pdf_title,
        )
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def build_base_metadata(
    path: Path,
    text: str,
    file_type: str,
    page: Optional[int] = None,
    data_root: Optional[Path] = None,
    doc_title: Optional[str] = None,
) -> dict:
    """
    构建通用文档元数据
    Args:
        path: 文档文件路径
        text: 文档内容
        file_type: 文档文件类型
        page: 文档页面编号（可选）
        data_root: 校园知识库根目录路径（可选）
        doc_title: 已确定的文档标题（可选）
    Returns:
        dict: 文档元数据字典
    """
    # 转换为绝对路径
    normalized_path = path.resolve()
    # 计算相对路径
    relative_path = _relative_path(normalized_path, data_root)
    # 文档标题
    doc_title = doc_title or infer_doc_title(path, text, file_type)
    # 文档分类
    doc_category = infer_doc_category(path)
    # 文档部门
    department = infer_department(text)
    # 文档ID种子
    doc_id_seed = f"{relative_path}#{page}" if page is not None else relative_path
    # 文档ID
    doc_id = hashlib.md5(doc_id_seed.encode("utf-8")).hexdigest()

    return {
        "source": str(path), # 文档原始路径
        "source_name": path.name, # 文档原始文件名
        "relative_path": relative_path, # 文档相对路径
        "file_type": file_type, # 文档文件类型
        "doc_title": doc_title, # 文档标题
        "doc_category": doc_category, # 文档分类
        "department": department, # 文档部门
        "doc_id": doc_id, # 文档ID
        "parent_id": doc_id, # 父文档ID
        "doc_type": "parent", # 文档类型
        "page": page, # 文档页面编号（可选）
        "content_length": len(text), # 文档内容长度
        "line_count": len(text.splitlines()), # 文档内容行数
        "file_size": path.stat().st_size if path.exists() else 0, # 文档文件大小（字节）
    }


def infer_doc_title(path: Path, text: str, file_type: str) -> str:
    """
    从正文或文件名推断标题
    Args:
        path: 文档文件路径
        text: 文档内容
        file_type: 文档文件类型
    Returns:
        str: 推断的标题
    """
    # 提取非空行文本为列表
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return path.stem or "未知文档"

    if file_type == "md":
        for line in lines[:10]:
            if line.startswith("#"):
                # 提取标题文本
                candidate = _clean_title(line)
                if candidate:
                    return candidate

    for line in lines[:5]:
        candidate = _clean_title(line)
        if _looks_like_title(candidate):
            return candidate
    # 返回文件名或第一行内容或默认值
    return path.stem or lines[0] or "未知文档"


def infer_doc_category(path: Path) -> str:
    """
    根据路径推断校园文档分类
    Args:
        path: 文档文件路径
    Returns:
        str: 推断的分类
    """
    # 将路径拆为多个部分并转换为小写，用于匹配分类
    path_parts = {part.lower() for part in path.parts}
    for key, value in CATEGORY_MAPPING.items():
        if key in path_parts:
            return value
    return FALLBACK_CATEGORY


def infer_department(text: str) -> str:
    """
    尝试从正文中提取发布部门
    Args:
        text: 文档内容
    Returns:
        str: 推断的部门
    """
    department_patterns = ("教务处", "学生处", "后勤处", "图书馆", "保卫处", "学院", "综合服务大厅")
    for pattern in department_patterns:
        if pattern in text:
            return pattern
    return ""


def normalize_text(text: str) -> str:
    """
    统一换行、空白和空行
    Args:
        text: 输入文本
    Returns:
        str: 统一后的文本
    """
    # 统一换行符为 \n
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    # 替换不间断空格为普通空格
    normalized = normalized.replace("\u00a0", " ")

    lines: List[str] = []
    # 上一行是否为空行
    blank_line_emitted = False
    for raw_line in normalized.split("\n"):
        # 将连续空格替换为单个空格，并移除行尾空白
        line = re.sub(r"[ \t]+", " ", raw_line).rstrip()
        if line.strip():
            lines.append(line.strip())
            blank_line_emitted = False
        elif not blank_line_emitted:
            lines.append("")
            blank_line_emitted = True

    return "\n".join(lines).strip()


def _read_text_file(path: Path) -> str:
    """
    读取文本文件内容，使用官方 TextLoader 逐个尝试常见编码
    Args:
        path: 文本文件路径
    Returns:
        str: 解码后的文本内容，或空字符串
    """
    for encoding in TEXT_ENCODINGS:
        try:
            # 使用官方 loader 读取文本
            loader = TextLoader(str(path), encoding=encoding)
            documents = loader.load()
            if documents:
                return documents[0].page_content
        except (RuntimeError, UnicodeDecodeError, UnicodeError, LookupError):
            continue
        except Exception as exc:  # pragma: no cover - 外部文件问题
            logger.warning("读取文本文件失败: %s, error=%s", path, exc)
            return ""

    logger.warning("文本文件编码无法识别: %s", path)
    return ""


def _clean_title(text: str) -> str:
    """
    清理标题文本，移除前导和尾随空格，以及数字和括号
    Args:
        text: 输入标题文本
    Returns:
        str: 清理后的标题文本
    """
    cleaned = re.sub(r"^[#\-\*\s]+", "", text).strip()
    cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned).strip()
    return cleaned


def _looks_like_title(text: str) -> bool:
    """
    判断文本是否像标题
    Args:
        text: 输入文本
    Returns:
        bool: 如果文本像标题则返回 True，否则返回 False
    """
    if not text:
        return False
    if len(text) > 80:
        return False
    # 以标点符号结尾的文本通常不是标题
    if text[-1] in "。.!！？?":
        return False
    # 以数字开头的文本通常不是标题
    if re.match(r"^\d+[\.\)]", text):
        return False
    return True


def _relative_path(path: Path, data_root: Optional[Path]) -> str:
    """
    计算相对路径
    Args:
        path: 输入路径
        data_root: 校园知识库根目录路径（可选）
    Returns:
        str: 相对路径
    """
    if data_root is not None:
        try:
            return path.relative_to(data_root.resolve()).as_posix()
        except Exception:
            pass
    return path.as_posix()
