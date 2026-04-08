import os
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from vectory_stores import VectoryService   # 如果你的实际文件名叫 vector_stores.py，这里也要改
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.runnables import RunnableLambda, RunnableWithMessageHistory
from langchain_core.documents import Document
from langchain_community.embeddings import DashScopeEmbeddings
from file_history_store import get_history


class RagService:
    def __init__(self):
        # 优先从 config_data 里取，其次从环境变量里取
        api_key = getattr(config, "DASHSCOPE_API_KEY", None) or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("未找到 DASHSCOPE_API_KEY，请先在 config_data.py 或环境变量中配置。")

        self.vector_service = VectoryService(
            embedding=DashScopeEmbeddings(
                model="text-embedding-v4",
                dashscope_api_key=api_key,
            )
        )

        self.prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                "以我提供的已知参考材料为主，简洁且专业地回答用户问题。若参考资料不足，就明确说明。参考资料如下：\n{context}"
            ),
            ("system", "以下是用户的历史对话："),
            MessagesPlaceholder(variable_name="history"),
            ("human", "请回答用户提问：{input}")
        ])

        self.chat_model = ChatTongyi(
            model="qwen3-max",
            dashscope_api_key=api_key,
        )

        self.chain = self.__get_chain()

    def __get_chain(self):
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]) -> str:
            if not docs:
                return "无相关参考资料"

            formatted = []
            for i, doc in enumerate(docs, start=1):
                formatted.append(
                    f"文档片段{i}：{doc.page_content}\n文档元数据：{doc.metadata}"
                )
            return "\n\n".join(formatted)

        # 这里必须显式保留 input / history，并且让 retriever 只吃 input 字段
        base_chain = (
            {
                "input": itemgetter("input"),
                "history": itemgetter("history"),
                "context": itemgetter("input") | retriever | RunnableLambda(format_document),
            }
            | self.prompt_template
            | self.chat_model
            | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            base_chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        return conversation_chain


if __name__ == "__main__":
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }

    rag = RagService()

    # 注意：这里要传 dict，不要直接传字符串
    res = rag.chain.invoke(
        {"input": ""},
        config=session_config
    )
    print(res)
