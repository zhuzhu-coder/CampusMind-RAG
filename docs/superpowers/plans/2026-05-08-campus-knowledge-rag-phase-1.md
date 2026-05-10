# 校园知识库 RAG 第一阶段实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把当前食谱 RAG 彻底换皮为支持 `pdf`、`md`、`txt` 的校园知识库 RAG，并保留 grounded answers、引用来源和检索评估。

**Architecture:** 先替换数据目录和公开语义，再把文件接入拆成独立的文档 ingest 层，最后同步改掉回答结构、检索评估和测试。第一版只支持可直接抽取文本的 PDF，不引入 OCR 或 FastAPI，目标是让本地索引、问答、eval 都能稳定跑通。

**Tech Stack:** Python 3.12, LangChain 1.x, FAISS, rank_bm25, pypdf, pytest

---

### Task 1: 切换默认数据路径和项目入口语义

**Files:**
- Modify: `code/config.py`
- Modify: `code/tests/test_config.py`
- Modify: `README.md`

- [ ] **Step 1: 写失败测试**

```python
from pathlib import Path

from config import PROJECT_ROOT, RAGConfig


def test_default_paths_are_absolute_and_project_relative():
    config = RAGConfig()

    assert Path(config.data_path).is_absolute()
    assert Path(config.index_save_path).is_absolute()
    assert Path(config.data_path) == PROJECT_ROOT / "data" / "campus"
    assert Path(config.index_save_path) == PROJECT_ROOT / "vector_index"
```

- [ ] **Step 2: 先跑测试确认失败**

Run: `cd E:\RAG\code; python -m pytest tests/test_config.py -q -p no:cacheprovider`

Expected: 失败，`data_path` 仍然指向 `data/cook`。

- [ ] **Step 3: 写最小实现**

```python
data_path: str = field(default_factory=lambda: str(PROJECT_ROOT / "data" / "campus"))
```

README 改成校园项目说明，标题和数据目录示例都改为 `Campus Knowledge RAG` 和 `data/campus`。

- [ ] **Step 4: 再跑测试确认通过**

Run: `cd E:\RAG\code; python -m pytest tests/test_config.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add code/config.py code/tests/test_config.py README.md
git commit -m "feat: switch default rag paths to campus"
```

---

### Task 2: 替换语料和烟雾评估集

**Files:**
- Delete: `data/cook/**`
- Create: `data/campus/regulations/学生请假管理办法.md`
- Create: `data/campus/teaching/期末考试安排.txt`
- Create: `data/campus/life/校园卡补办说明.md`
- Create: `data/campus/notices/图书馆临时闭馆.pdf`
- Delete: `code/evals/recipe_eval_set.jsonl`
- Create: `code/evals/campus_smoke_eval_set.jsonl`
- Modify: `code/tests/test_data_preparation_smoke.py`

- [ ] **Step 1: 写失败测试**

```python
from config import PROJECT_ROOT


def test_campus_sample_corpus_exists():
    root = PROJECT_ROOT / "data" / "campus"

    assert root.exists()
    assert list(root.rglob("*.md"))
    assert list(root.rglob("*.txt"))
    assert list(root.rglob("*.pdf"))
```

- [ ] **Step 2: 先跑测试确认失败**

Run: `cd E:\RAG\code; python -m pytest tests/test_data_preparation_smoke.py -q -p no:cacheprovider`

Expected: 失败，因为 `data/campus` 还不存在，旧的 `data/cook` 也还没有替换。

- [ ] **Step 3: 写最小实现**

把语料换成能直接支撑校园问答的最小集合。建议每个文件都覆盖一个真实问题点，例如：

```md
# 学生请假管理办法

## 请假审批

学生请假超过三天需由辅导员和学院审批。
```

```txt
期末考试期间，如需缓考，请在教务系统提交申请。
```

```json
{"id":"campus-001","question":"学生请假超过三天需要谁审批？","expected_doc_titles":["学生请假管理办法"],"expected_keywords":["辅导员","学院"],"category":"规章制度"}
```

PDF 只要求是可抽取文本的普通 PDF，不做 OCR。

- [ ] **Step 4: 再跑测试确认通过**

Run: `cd E:\RAG\code; python -m pytest tests/test_data_preparation_smoke.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add data/campus code/evals/campus_smoke_eval_set.jsonl code/tests/test_data_preparation_smoke.py
git rm -r data/cook code/evals/recipe_eval_set.jsonl
git commit -m "feat: replace recipe corpus with campus corpus"
```

---

### Task 3: 抽出文档接入层，支持 pdf/md/txt

**Files:**
- Create: `code/rag_modules/document_ingestion.py`
- Modify: `code/requirements.txt`
- Create: `code/tests/test_document_ingestion.py`

- [ ] **Step 1: 写失败测试**

```python
from config import PROJECT_ROOT
from rag_modules.document_ingestion import load_documents


def test_load_documents_returns_supported_file_types():
    docs = load_documents(PROJECT_ROOT / "data" / "campus")

    assert docs
    assert {doc.metadata["file_type"] for doc in docs} >= {"md", "txt", "pdf"}
    assert all(doc.metadata["doc_title"] for doc in docs)
```

- [ ] **Step 2: 先跑测试确认失败**

Run: `cd E:\RAG\code; python -m pytest tests/test_document_ingestion.py -q -p no:cacheprovider`

Expected: 失败，因为 `rag_modules.document_ingestion` 还不存在。

- [ ] **Step 3: 写最小实现**

```python
from pathlib import Path
from langchain_core.documents import Document
from pypdf import PdfReader


SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


def load_documents(data_path: str | Path) -> list[Document]:
    root = Path(data_path)
    documents = []
    for path in sorted(root.rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".md":
            documents.append(load_markdown_document(path))
        elif suffix == ".txt":
            documents.append(load_text_document(path))
        elif suffix == ".pdf":
            documents.extend(load_pdf_document(path))
    return documents


def load_pdf_document(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    documents = []
    for page_number, page in enumerate(reader.pages, 1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue
        metadata = build_base_metadata(path, text, file_type="pdf", page=page_number)
        documents.append(Document(page_content=text, metadata=metadata))
    return documents
```

实现时把公共能力拆成几个小函数：

- `normalize_text`
- `infer_doc_title`
- `load_markdown_document`
- `load_text_document`
- `load_pdf_document`
- `load_documents`

`load_pdf_document` 只处理可直接抽取文本的 PDF；空页记录日志，整份 PDF 没文本就跳过并标记为不支持。

- [ ] **Step 4: 再跑测试确认通过**

Run: `cd E:\RAG\code; python -m pytest tests/test_document_ingestion.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add code/rag_modules/document_ingestion.py code/requirements.txt code/tests/test_document_ingestion.py
git commit -m "feat: add campus document ingestion"
```

---

### Task 4: 重构 DataPreparationModule 以适配校园文档

**Files:**
- Modify: `code/rag_modules/data_preparation.py`
- Modify: `code/tests/test_data_preparation_smoke.py`

- [ ] **Step 1: 写失败测试**

```python
from config import PROJECT_ROOT
from rag_modules.data_preparation import DataPreparationModule


def test_campus_documents_can_be_loaded_and_chunked():
    module = DataPreparationModule(str(PROJECT_ROOT / "data" / "campus"))
    documents = module.load_documents()
    chunks = module.chunk_documents()
    stats = module.get_statistics()

    assert documents
    assert chunks
    assert stats["total_documents"] >= 1
    assert "规章制度" in stats["categories"]
    assert all("doc_title" in doc.metadata for doc in documents)
```

- [ ] **Step 2: 先跑测试确认失败**

Run: `cd E:\RAG\code; python -m pytest tests/test_data_preparation_smoke.py -q -p no:cacheprovider`

Expected: 失败，因为 `DataPreparationModule` 还在直接读 Markdown 食谱，并依赖 `dish_name` / `category` / `difficulty`。

- [ ] **Step 3: 写最小实现**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .document_ingestion import load_documents as load_campus_documents


def load_documents(self) -> list[Document]:
    self.documents = load_campus_documents(self.data_path)
    return self.documents
```

分块逻辑要按文件类型分流：

- `md`：有标题时用标题切分，没标题就退回通用分块
- `txt`：先按段落，再按长度
- `pdf`：按页继续切分

元数据改成校园字段：

- `doc_title`
- `doc_category`
- `department`
- `file_type`
- `source`
- `section`
- `page`
- `doc_id`
- `chunk_id`

`get_statistics()` 也要从 `category` / `difficulty` 改成校园知识库统计。

- [ ] **Step 4: 再跑测试确认通过**

Run: `cd E:\RAG\code; python -m pytest tests/test_data_preparation_smoke.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add code/rag_modules/data_preparation.py code/tests/test_data_preparation_smoke.py
git commit -m "feat: refactor data preparation for campus docs"
```

---

### Task 5: 切换回答结构、Prompt 和主流程到校园语义

**Files:**
- Modify: `code/rag_modules/response_schema.py`
- Modify: `code/rag_modules/generation_integration.py`
- Modify: `code/main.py`
- Modify: `code/tests/test_response_schema.py`
- Modify: `code/tests/test_recipe_rag_system.py`

- [ ] **Step 1: 写失败测试**

```python
from rag_modules import RAGResponse, RetrievedSource


def test_rag_response_serializes_sources_to_plain_dicts():
    source = RetrievedSource(
        source_id=1,
        doc_title="学生请假管理办法",
        doc_category="规章制度",
        department="学生处",
        file_type="md",
        section="请假审批",
        source="data/campus/regulations/学生请假管理办法.md",
        page=None,
        chunk_index=2,
        rrf_score=0.0325,
        snippet="学生请假超过三天需由辅导员和学院审批。",
    )
```

```python
source = response.sources[0]
assert source.doc_title == "学生请假管理办法"
assert source.doc_category == "规章制度"
assert source.file_type == "md"
```

- [ ] **Step 2: 先跑测试确认失败**

Run: `cd E:\RAG\code; python -m pytest tests/test_response_schema.py tests/test_recipe_rag_system.py -q -p no:cacheprovider`

Expected: 失败，因为现在的 schema 还是 `dish_name` / `difficulty`，主流程还是 `RecipeRAGSystem`。

- [ ] **Step 3: 写最小实现**

```python
@dataclass(frozen=True)
class RetrievedSource:
    source_id: int
    doc_title: str
    doc_category: str
    department: str
    file_type: str
    section: str
    source: str
    page: Optional[int]
    chunk_index: int
    rrf_score: Optional[float]
    snippet: str
```

```text
你是校园知识库问答助手。
只能基于检索到的校园文档回答。
不能编造制度、日期、地点或联系方式。
信息不足时，明确说“当前资料不足”。
```

`main.py` 里需要同步做这些事：

- 把公开类名从 `RecipeRAGSystem` 改成 `CampusRAGSystem`
- 删除或重命名所有食谱专用 helper
- 把 `ask_question(question, return_sources=True)` 保持成结构化输出入口
- 把打印文案从“菜谱”改成“校园文档”

- [ ] **Step 4: 再跑测试确认通过**

Run: `cd E:\RAG\code; python -m pytest tests/test_response_schema.py tests/test_recipe_rag_system.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add code/rag_modules/response_schema.py code/rag_modules/generation_integration.py code/main.py code/tests/test_response_schema.py code/tests/test_recipe_rag_system.py
git commit -m "feat: switch rag response and main flow to campus"
```

---

### Task 6: 切换检索评估到校园 smoke 集

**Files:**
- Modify: `code/evals/run_retrieval_eval.py`
- Delete: `code/evals/recipe_eval_set.jsonl`
- Create: `code/evals/campus_smoke_eval_set.jsonl`
- Modify: `code/tests/test_retrieval_eval.py`

- [ ] **Step 1: 写失败测试**

```python
from langchain_core.documents import Document
from evals.run_retrieval_eval import extract_doc_titles, reciprocal_rank


def test_extract_doc_titles_prefers_metadata_over_content():
    docs = [
        Document(page_content="# 学生请假管理办法\n请假审批流程", metadata={"doc_title": "学生请假管理办法"}),
        Document(page_content="# 图书馆借阅规则\n借阅期限说明", metadata={}),
    ]

    assert extract_doc_titles(docs) == ["学生请假管理办法", "图书馆借阅规则"]
```

- [ ] **Step 2: 先跑测试确认失败**

Run: `cd E:\RAG\code; python -m pytest tests/test_retrieval_eval.py -q -p no:cacheprovider`

Expected: 失败，因为现在还是 `extract_dish_names` 和 `expected_dishes`。

- [ ] **Step 3: 写最小实现**

```python
DEFAULT_EVAL_SET = Path(__file__).resolve().with_name("campus_smoke_eval_set.jsonl")


def extract_doc_titles(docs: Iterable[Document]) -> List[str]:
    doc_titles = []
    seen = set()
    for doc in docs:
        metadata = doc.metadata or {}
        doc_title = metadata.get("doc_title")
        if not doc_title:
            first_line = (doc.page_content or "").strip().splitlines()[0:1]
            doc_title = first_line[0].replace("#", "").strip() if first_line else ""
        doc_title = doc_title or "未知文档"
        if doc_title in seen:
            continue
        doc_titles.append(doc_title)
        seen.add(doc_title)
    return doc_titles
```

`campus_smoke_eval_set.jsonl` 的每条记录改成围绕校园知识库的问题，字段改为：

- `id`
- `question`
- `expected_doc_titles`
- `expected_keywords`
- `category`

`main()` 默认读取新评估集，`--strategies` 仍然保留 `vector`、`bm25`、`hybrid`。

- [ ] **Step 4: 再跑测试确认通过**

Run: `cd E:\RAG\code; python -m pytest tests/test_retrieval_eval.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add code/evals/run_retrieval_eval.py code/evals/campus_smoke_eval_set.jsonl code/tests/test_retrieval_eval.py
git rm code/evals/recipe_eval_set.jsonl
git commit -m "feat: add campus retrieval eval"
```

---

### Task 7: 清理 README 和做最终验收

**Files:**
- Modify: `README.md`
- Delete: `data/cook/**` if any leftovers remain

- [ ] **Step 1: 做一次全仓库残留检查**

Run: `cd E:\RAG; rg -n "Recipe|recipe|食谱|菜谱|dish_name|difficulty|ingredients|data/cook" README.md code data`

Expected: 只剩下历史提交里才会出现的内容，工作区内不应再有这些词。

- [ ] **Step 2: 把 README 改成校园项目说明**

```md
# 校园知识库 RAG

支持校园规章制度、教务教学、生活通知等文档问答，默认读取 `data/campus`。
```

运行、测试、评估三段都改成校园项目语义。

- [ ] **Step 3: 做最终回归验证**

Run:

```powershell
cd E:\RAG\code
python -m pytest tests -q -p no:cacheprovider
python evals/run_retrieval_eval.py --json
python main.py
```

Expected:

- pytest 全绿
- eval 能输出校园检索指标
- CLI 能正常初始化并返回校园知识库答案

- [ ] **Step 4: 提交**

```powershell
git add README.md data/campus code
git rm -r data/cook
git commit -m "feat: complete campus knowledge rag phase 1"
```
