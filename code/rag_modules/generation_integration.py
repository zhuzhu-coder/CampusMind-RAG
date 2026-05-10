"""
生成集成模块
"""

import os
import logging
import re
from typing import Iterable, Iterator, List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        # 初始化模型名称
        self.model_name = model_name
        # 初始化生成温度
        self.temperature = temperature
        # 初始化最大token数
        self.max_tokens = max_tokens
        # 初始化LLM模型
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
        """初始化LLM模型"""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
        # 初始化LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )
        logger.info("LLM初始化完成")
    
    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式
        Args:
            query: 用户查询
        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取文档列表或推荐，只需要文档标题
   例如：有哪些学生管理规定、推荐几份期末安排、给我3个校园通知

2. 'detail' - 用户想要具体办理方法、流程或详细信息
   例如：学生请假超过三天怎么审批、缓考怎么申请、补办校园卡要哪些材料

3. 'general' - 其他一般性问题
   例如：什么是请假管理办法、校园卡补办是什么意思、考试安排怎么看

请只返回分类结果：list、detail 或 general

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query)
        # 规范化路由类型
        return self._normalize_route_type(result)

    @staticmethod
    def _normalize_route_type(result: str) -> str:
        """
        从模型输出中提取稳定的路由类型
        Args:
            result: 模型原始输出
        Returns:
            list / detail / general
        """
        # 规范化输入，转换为小写并移除首尾空格
        normalized = (result or "").strip().lower()
        # 检查是否包含路由类型
        for route_type in ("list", "detail", "general"):
            if re.search(rf"(?<![a-z]){route_type}(?![a-z])", normalized):
                return route_type
        return "general"

    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询
        Args:
            query: 原始查询
        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate.from_template("""
你是一个智能查询分析助手。请分析用户的查询，判断是否需要重写以提高校园文档检索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体文档标题：如"学生请假管理办法怎么规定"、"期末考试安排在哪里看"
   - 明确的办理询问：如"校园卡补办需要什么材料"、"缓考申请的步骤"
   - 具体的规则问题：如"请假超过三天谁审批"、"考试证件要带什么"

2. **模糊不清的查询**（需要重写）：
   - 过于宽泛：如"请假"、"考试"、"校园卡"
   - 缺乏具体信息：如"管理办法"、"通知"、"流程"
   - 口语化表达：如"怎么办"、"有什么要求"、"怎么弄"

重写原则：
- 保持原意不变
- 增加相关校园文档检索术语
- 优先补足问题中缺失的对象、流程或条件
- 保持简洁性

示例：
- "请假" → "学生请假管理办法"
- "有通知吗" → "校园通知公告"
- "推荐个文件" → "校园文档推荐"
- "校园卡" → "校园卡补办说明"
- "学生请假超过三天需要谁审批" → "学生请假超过三天需要谁审批"（保持原查询）
- "缓考申请的步骤" → "缓考申请的步骤"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response

    def _build_context(self, parent_docs: List[Document], max_length: int = 2000) -> str:
        """
        构建上下文字符串
        Args:
            parent_docs: 用于生成回答的完整父文档列表
            max_length: 最大长度
        Returns:
            格式化的上下文字符串
        """
        if not parent_docs:
            return "暂无相关校园文档信息。"

        separator = "\n" + "=" * 50 + "\n"

        context_parts = [] # 保存每个父文档格式化后的文本
        current_length = len(separator) # 已占用的字符长度
        # 遍历每个父文档，格式化并添加到上下文
        for i, parent_doc in enumerate(parent_docs, 1):
            doc_text = self._format_context_doc(i, parent_doc, include_optional_metadata=True)

            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                # 尝试不包含可选元数据的格式化
                compact_doc_text = self._format_context_doc(i, parent_doc, include_optional_metadata=False)
                if len(compact_doc_text) < len(doc_text):
                    doc_text = compact_doc_text
                remaining_length = max_length - current_length
                # 判断是否还有剩余长度
                if remaining_length > 0:
                    # 截断文档文本，确保不超过最大长度
                    suffix = "..." if remaining_length > 3 else ""
                    truncated_text = doc_text[:remaining_length - len(suffix)].rstrip()
                    context_parts.append(truncated_text + suffix)
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        return (separator + "\n".join(context_parts))[:max_length]

    def _format_context_doc(
        self,
        source_id: int,
        parent_doc: Document,
        include_optional_metadata: bool = True,
    ) -> str:
        """
        将单个完整父文档格式化为带引用编号的上下文文本
        Args:
            source_id: 来源ID
            parent_doc: 完整父文档
            include_optional_metadata: 是否包含可选元数据
        Returns:
            格式化的文档字符串
        """
        metadata = parent_doc.metadata or {}
        doc_title = GenerationIntegrationModule._get_doc_title(metadata)
        metadata_info = f"[{source_id}] 校园文档: {doc_title}"

        if include_optional_metadata:
            if metadata.get("doc_category"):
                metadata_info += f" | 分类: {metadata['doc_category']}"
            if metadata.get("department"):
                metadata_info += f" | 部门: {metadata['department']}"
            if metadata.get("file_type"):
                metadata_info += f" | 类型: {metadata['file_type']}"
        if metadata.get("source"):
            metadata_info += f"\n来源: {metadata['source']}"

        return f"{metadata_info}\n内容:\n{parent_doc.page_content}\n"

    @staticmethod
    def _grounded_answer_rules() -> str:
        """生成回答时使用的检索约束和引用规则。"""
        return """
回答规则：
1. 只能基于“相关校园文档信息”回答，不要使用未检索到的外部知识。
2. 不要编造制度、流程、日期、地点、材料或来源。
3. 如果相关校园文档信息不足以回答问题，明确说明“当前资料不足”，不要猜测。
4. 每个关键要求、步骤、提示或结论后面必须标注来源编号，例如 [1]。
5. 参考来源列表会由系统自动追加，不要自行编写参考来源小节。
"""

    @staticmethod
    def _build_reference_lines(parent_docs: List[Document]) -> List[str]:
        """
        根据上下文父文档生成参考来源行
        Args:
            parent_docs: 用于生成回答的完整父文档列表
        Returns:
            参考来源行列表
        """
        reference_lines = []
        seen_titles = set()
        for source_id, parent_doc in enumerate(parent_docs, 1):
            metadata = parent_doc.metadata or {}
            doc_title = GenerationIntegrationModule._get_doc_title(metadata)
            if doc_title in seen_titles:
                continue
            line = f"[{source_id}] {doc_title}"
            if metadata.get("source"):
                line += f" - {metadata['source']}"
            reference_lines.append(line)
            seen_titles.add(doc_title)
        return reference_lines

    def _append_reference_lines(self, answer: str, parent_docs: List[Document]) -> str:
        """
        在答案末尾追加稳定的参考来源列表
        Args:
            answer: 模型生成的答案
            parent_docs: 用于生成回答的完整父文档列表
        Returns:
            追加参考来源后的答案
        """
        if "参考来源" in answer:
            return answer

        reference_lines = self._build_reference_lines(parent_docs)
        if not reference_lines:
            return answer

        return answer.rstrip() + "\n\n参考来源:\n" + "\n".join(reference_lines)

    def _stream_with_reference_lines(
        self,
        text_chunks: Iterable[str],
        parent_docs: List[Document],
    ) -> Iterator[str]:
        """
        流式输出模型文本，并在末尾按需追加稳定的参考来源列表
        Args:
            text_chunks: 模型流式生成的文本片段
            parent_docs: 用于生成回答的完整父文档列表
        Yields:
            模型文本片段，以及可能追加的参考来源片段
        """
        generated_parts = []
        for text_chunk in text_chunks:
            generated_parts.append(text_chunk)
            yield text_chunk

        generated_text = "".join(generated_parts)
        if "参考来源" in generated_text:
            return

        reference_lines = self._build_reference_lines(parent_docs)
        if reference_lines:
            yield "\n\n参考来源:\n" + "\n".join(reference_lines)

    def generate_list_answer(self, query: str, parent_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询
        Args:
            query: 用户查询
            parent_docs: 用于生成回答的完整父文档列表
        Returns:
            列表式回答
        """
        if not parent_docs:
            return "抱歉，没有找到相关的校园文档。"
        # 构造文档引用列表
        doc_refs = []
        seen_doc_titles = set()
        for source_id, parent_doc in enumerate(parent_docs, 1):
            doc_title = self._get_doc_title(parent_doc.metadata or {})
            if doc_title not in seen_doc_titles:
                doc_refs.append((doc_title, source_id))
                seen_doc_titles.add(doc_title)

        # 构建简洁的列表回答
        if len(doc_refs) == 1:
            answer = f"为您推荐：{doc_refs[0][0]} [{doc_refs[0][1]}]"
        elif len(doc_refs) <= 3:
            answer = "为您推荐以下文档：\n" + "\n".join(
                [f"{i+1}. {name} [{source_id}]" for i, (name, source_id) in enumerate(doc_refs)]
            )
        else:
            answer = "为您推荐以下文档：\n" + "\n".join(
                [f"{i+1}. {name} [{source_id}]" for i, (name, source_id) in enumerate(doc_refs[:3])]
            )
            answer += f"\n\n还有其他 {len(doc_refs)-3} 份文档可供选择。"
        # 追加参考来源
        reference_lines = self._build_reference_lines(parent_docs)
        if reference_lines:
            answer += "\n\n参考来源:\n" + "\n".join(reference_lines)
        return answer

    def generate_basic_answer(self, query: str, parent_docs: List[Document]) -> str:
        """
        生成基础回答
        Args:
            query: 用户查询
            parent_docs: 用于生成回答的完整父文档列表
        Returns:
            生成的回答
        """
        context = self._build_context(parent_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的校园知识库助手。请根据以下校园文档信息回答用户的问题。

用户问题: {question}

相关校园文档信息:
{context}

""" + self._grounded_answer_rules() + """

请在遵守以上规则的前提下，提供详细、实用的回答。

回答:""")

        # 构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context} # 可运行的映射规则
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return self._append_reference_lines(response, parent_docs)

    def generate_step_by_step_answer(self, query: str, parent_docs: List[Document]) -> str:
        """
        生成分步骤详细回答
        Args:
            query: 用户查询
            parent_docs: 用于生成回答的完整父文档列表
        Returns:
            分步骤的详细回答
        """
        context = self._build_context(parent_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的校园事务助手。请根据校园文档信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关校园文档信息:
{context}

""" + self._grounded_answer_rules() + """

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 文档概览
[简要介绍文档主题和适用范围]

## 关键要求
[列出主要条件、要求或材料]

## 办理步骤
[详细的分步骤说明，每步包含具体操作和注意事项]

## 注意事项
[仅在有实用提醒时包含。优先使用原文中的要求与提示，必要时可以基于正文总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的注意事项要分享，可以省略该部分

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return self._append_reference_lines(response, parent_docs)


    def generate_basic_answer_stream(self, query: str, parent_docs: List[Document]):
        """
        生成基础回答 - 流式输出
        Args:
            query: 用户查询
            parent_docs: 用于生成回答的完整父文档列表
        Yields:
            生成的回答片段
        """
        context = self._build_context(parent_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的校园知识库助手。请根据以下校园文档信息回答用户的问题。

用户问题: {question}

相关校园文档信息:
{context}

""" + self._grounded_answer_rules() + """

请在遵守以上规则的前提下，提供详细、实用的回答。

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        yield from self._stream_with_reference_lines(chain.stream(query), parent_docs)

    @staticmethod
    def _get_doc_title(metadata: dict) -> str:
        """从元数据中提取稳定的文档标题"""
        return (
            metadata.get("doc_title")
            or metadata.get("source_name")
            or "未知文档"
        )

    def generate_step_by_step_answer_stream(self, query: str, parent_docs: List[Document]):
        """
        生成详细步骤回答 - 流式输出
        Args:
            query: 用户查询
            parent_docs: 用于生成回答的完整父文档列表
        Yields:
            详细步骤回答片段
        """
        context = self._build_context(parent_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的校园事务助手。请根据校园文档信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关校园文档信息:
{context}

""" + self._grounded_answer_rules() + """

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 文档概览
[简要介绍文档主题和适用范围]

## 关键要求
[列出主要条件、要求或材料]

## 办理步骤
[详细的分步骤说明，每步包含具体操作和注意事项]

## 注意事项
[仅在有实用提醒时包含。如果原文内容与办理无关或为空，可以基于正文总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容
- 重点突出实用性和可操作性

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        yield from self._stream_with_reference_lines(chain.stream(query), parent_docs)
