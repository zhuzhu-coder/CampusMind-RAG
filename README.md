# 校园知识库 RAG

这是一个基于 LangChain 1.x 的校园知识库 RAG 示例项目。系统会读取校园文档，构建 FAISS 向量索引，并结合 BM25 与向量检索回答校园相关问题，同时保留 grounded answers 和引用来源。

## 目录结构

```text
code/
  main.py
  config.py
  rag_modules/
  tests/
data/
  campus/
```

默认读取 `data/campus` 下的文档，索引默认保存到 `vector_index`。路径由 `config.py` 按项目根目录解析，所以从项目根或 `code/` 目录运行都能得到一致结果。

## 安装

```powershell
cd E:\RAG\code
python -m pip install -r requirements.txt
```

## 配置

复制 `.env.example` 为 `.env`，然后填入 DashScope API Key。

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

可选的 `RAG_*` 配置项：

```env
RAG_DATA_PATH=data/campus
RAG_INDEX_SAVE_PATH=vector_index
RAG_EMBEDDING_MODEL=text-embedding-v4
RAG_LLM_MODEL=qwen3.5-plus
RAG_TOP_K=3
RAG_RETRIEVAL_CANDIDATE_K=10
RAG_TEMPERATURE=0.1
RAG_MAX_TOKENS=2048
```

## 准备数据

把校园文档放到 `data/campus` 目录，建议按分类子目录组织，例如：

```text
data/campus/regulations/student_affairs/学生请假管理办法.md
data/campus/teaching/exams/期末考试安排.txt
```

## 运行

```powershell
cd E:\RAG\code
python main.py
```

## 测试

```powershell
cd E:\RAG\code
python -m pytest tests -q -p no:cacheprovider
```

## 校园检索评估

项目内置了一个轻量评估集，用于对比向量检索、BM25 和 Hybrid+RRF 的召回效果。默认评估集是 `code/evals/campus_smoke_eval_set.jsonl`。

```powershell
cd E:\RAG\code
python evals/run_retrieval_eval.py
```

默认输出 `hit@1`、`hit@3` 和 `mrr`。如需机器可读结果：

```powershell
python evals/run_retrieval_eval.py --json
```
