# 校园知识库 RAG

面向校园规章、教务通知、生活服务、图书馆公告和网络维护等文档的 RAG 问答系统。项目基于 LangChain 1.x、FAISS、BM25 和 FastAPI，支持 PDF、Markdown、TXT 文档接入，提供命令行问答、Web 问答页、HTTP API、检索评估和结构化 trace。

这个仓库的目标不是只跑一个演示，而是沉淀一个可以验证、调优、集成的校园知识库后端。

## 核心能力

- 文档接入：加载 PDF、Markdown、TXT，并抽取标题、分类、部门、文件类型、来源路径、页码和章节等元数据。
- 索引缓存：使用 FAISS 保存向量索引，用 manifest 记录数据指纹、embedding 模型和分块策略，避免重复构建。
- 混合检索：支持 Vector、BM25、Hybrid 检索，并通过 RRF 融合重排。
- 证据窗口：检索命中 chunk 后，只回填相邻片段作为生成上下文，降低上下文噪声和 token 成本。
- Grounded Answer：约束回答基于检索上下文，并输出可追踪的来源编号。
- 服务接口：提供 CLI、FastAPI API、轻量 Web 页面、健康检查、预热、统计和问答接口。
- 可观测性：`sources[]` 返回结构化来源，`trace` 返回检索策略、耗时、参数、召回块和证据窗口摘要。
- 可评估性：内置 38 条校园问答评估集，当前 Hybrid 检索达到 `hit@1=0.9474`、`hit@3=1.0000`、`MRR=0.9737`。

## 技术栈

| 层级 | 选型 |
| --- | --- |
| 文档处理 | LangChain document loaders、pypdf |
| 分块 | MarkdownHeaderTextSplitter、RecursiveCharacterTextSplitter |
| 向量索引 | FAISS |
| 关键词检索 | BM25、jieba 中文分词 |
| 生成模型接入 | DashScope OpenAI-compatible API |
| API 服务 | FastAPI、Uvicorn |
| 测试与评估 | pytest、自建 JSONL 评估集 |

## 项目结构

```text
src/campus_rag/
  api.py                 # FastAPI 应用与路由
  cli.py                 # 交互式命令行入口
  config.py              # 环境变量与路径配置
  system.py              # RAG 系统编排
  pipeline/              # 文档接入、分块、索引、检索、生成、响应 schema
  static/                # 轻量 Web 问答页
tests/                   # 单元测试与 API 测试
evals/                   # 检索评估脚本、评估集和基线结果
data/campus/             # 校园知识库文档
pyproject.toml           # 包元数据、依赖、pytest 配置、命令入口
.env.example             # 环境变量模板
```

默认读取 `data/campus`，默认把本地索引写入 `vector_index/`。所有相对路径都会按项目根目录解析，因此从根目录、`src/` 子目录或安装后的包入口运行都能得到一致路径。

## 快速开始

### 1. 安装依赖

```powershell
cd E:\RAG
python -m pip install -e ".[dev]"
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并填入 DashScope API Key：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

常用可选配置：

```env
RAG_DATA_PATH=data/campus
RAG_INDEX_SAVE_PATH=vector_index
RAG_EMBEDDING_MODEL=text-embedding-v4
RAG_LLM_MODEL=qwen3.5-plus
RAG_TOP_K=3
RAG_RETRIEVAL_CANDIDATE_K=10
RAG_RRF_K=60
RAG_CONTEXT_WINDOW_SIZE=1
RAG_TEMPERATURE=0.1
RAG_MAX_TOKENS=2048
```

### 3. 准备数据

把校园文档放入 `data/campus/`，建议按业务域分层：

```text
data/campus/regulations/student_affairs/学生请假管理办法.md
data/campus/teaching/exams/期末考试安排.txt
data/campus/life/dormitory/宿舍晚归登记说明.md
```

### 4. 启动问答

命令行问答：

```powershell
python -m campus_rag.cli
```

或者使用安装后的命令：

```powershell
campus-rag
```

FastAPI 服务：

```powershell
python -X utf8 -m uvicorn campus_rag.api:app --host 127.0.0.1 --port 8000
```

启动后访问：

- Web 问答页：`http://127.0.0.1:8000/`
- Swagger 文档：`http://127.0.0.1:8000/docs`

Windows 本地运行建议保留 `-X utf8`，避免中文输出和部分终端编码问题。

## 系统流程

```text
校园文档 PDF/MD/TXT
        |
        v
文档加载与元数据抽取
        |
        v
结构化分块与 parent_id/chunk_index 标记
        |
        v
FAISS 向量索引 + BM25 关键词索引
        |
        v
Vector / BM25 / Hybrid 检索
        |
        v
RRF 重排与去重
        |
        v
命中 chunk 邻域证据窗口
        |
        v
Grounded Answer 生成
        |
        v
answer + sources[] + optional trace
```

证据窗口默认取命中 chunk 前后各 1 个相邻片段：

```text
chunk_index - window_size
chunk_index
chunk_index + window_size
```

可以通过 `RAG_CONTEXT_WINDOW_SIZE` 调整窗口大小。

## API

| Method | Path | 说明 |
| --- | --- | --- |
| GET | `/health` | 进程健康检查，不触发 RAG 初始化 |
| GET | `/ready` | 查看 RAG 是否已完成初始化 |
| POST | `/warmup` | 主动加载模型、文档和索引 |
| GET | `/stats` | 返回知识库文档、chunk、分类、部门、文件类型统计 |
| POST | `/ask` | 提交问题并返回答案、来源和可选 trace |

### 问答请求

```json
{
  "question": "我晚归了会怎么样",
  "return_sources": true,
  "return_trace": true
}
```

### 问答响应

```json
{
  "question": "我晚归了会怎么样",
  "route_type": "detail",
  "rewritten_query": "学生晚归后果及处理规定",
  "answer": "...",
  "sources": [
    {
      "source_id": 1,
      "doc_title": "宿舍晚归登记说明",
      "doc_category": "校园生活",
      "department": "学生处",
      "file_type": "md",
      "section": "晚归登记",
      "source": "data/campus/life/dormitory/宿舍晚归登记说明.md",
      "page": null,
      "chunk_index": 0,
      "rrf_score": 0.03,
      "snippet": "..."
    }
  ],
  "trace": {
    "retrieval_strategy": "hybrid",
    "filters": {},
    "timings_ms": {
      "analysis": 12.3,
      "retrieval": 8.1,
      "context_build": 0.5,
      "generation": 1200.0,
      "total": 1221.0
    },
    "retrieval_params": {
      "top_k": 3,
      "candidate_k": 10,
      "rrf_k": 60,
      "context_window_size": 1
    },
    "retrieved_chunks": [],
    "context_documents": [],
    "source_count": 1
  }
}
```

`return_sources=false` 时，响应中的 `sources` 会返回空数组。`return_trace=true` 时，即使隐藏来源，`trace.source_count` 仍会记录内部构建出的来源数量，便于调试。

## 检索评估

评估集位于 `evals/campus_smoke_eval_set.jsonl`，覆盖规章、教务、生活服务、图书馆和信息中心通知等场景。

BM25-only 评估不依赖 DashScope API Key：

```powershell
python evals\run_retrieval_eval.py --strategies bm25 --json
```

当前已保存 BM25 baseline：

```text
evals/results/2026-05-12-bm25-baseline.json
```

当前三策略对比 summary：

```text
evals/results/2026-05-12-vector-bm25-hybrid-summary.json
```

| Strategy | Cases | hit@1 | hit@3 | MRR | keyword_coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| vector | 38 | 0.8684 | 0.9737 | 0.9167 | 0.9342 |
| bm25 | 38 | 0.8684 | 1.0000 | 0.9342 | 0.9342 |
| hybrid | 38 | 0.9474 | 1.0000 | 0.9737 | 0.9605 |

Vector/Hybrid 评估需要可用的 DashScope API Key：

```powershell
python evals\run_retrieval_eval.py --strategies vector bm25 hybrid --json
```

## 测试

```powershell
python -m pytest
```

测试覆盖配置读取、文档加载、文档分块、索引 manifest、BM25/Hybrid 检索、RRF、证据窗口、评估指标、API 响应、trace 序列化、包入口和静态资源托管。

## 开发说明

- `.env`、`.venv/`、`vector_index/`、缓存目录和临时评估结果不会提交。
- `vector_index/` 是本地运行产物，删除后会在下次构建知识库时重新生成。
- `evals/results/2026-05-12-bm25-baseline.json` 是已验证 baseline，保留在仓库中便于对比。
- 修改检索、分块、来源或 trace 逻辑后，建议同时运行 `python -m pytest` 和 BM25/Hybrid 评估。
