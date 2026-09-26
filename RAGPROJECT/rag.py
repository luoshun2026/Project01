from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory, RunnableLambda
from file_history_store import get_history
from vector_stores import VectorStoreService
from langchain_community.embeddings import DashScopeEmbeddings
import config_data as config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models.tongyi import ChatTongyi


def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)

    return prompt


class RagService(object):
    def __init__(self):

        self.vector_service = VectorStoreService(
            embedding=DashScopeEmbeddings(model=config.embedding_model_name)
        )

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "你是专业的数据库与SQL知识点助手，严格基于以下提供的参考资料回答用户问题。\n"
                 "参考资料：{context}\n\n"
                 "回答要求：\n"
                 "1. 只使用参考资料中的内容，禁止编造知识点，资料中没有的请明确说明「暂无相关知识点」\n"
                 "2. 涉及SQL语法时请单独成行并格式化，关键语法高亮说明\n"
                 "3. 回答条理清晰，分点阐述，准确专业\n"
                 "4. 回答末尾标注参考来源，格式：【参考：xxx.txt】"),
                MessagesPlaceholder("history"),
                ("user", "用户提问：{input}")
            ]
        )

        self.chat_model = ChatTongyi(model=config.chat_model_name)

        self.chain = self.__get_chain()

    def __get_chain(self):
        """获取最终的执行链"""
        retriever = self.vector_service.get_retriever()

        def format_document(docs: list[Document]):
            if not docs:
                return "无相关参考资料"
            formatted_str = ""
            for idx, doc in enumerate(docs, 1):
                source = doc.metadata.get("source", "未知文档")
                category = doc.metadata.get("category", "未分类")
                formatted_str += f"【参考{idx} | 分类：{category} | 来源：{source}】\n{doc.page_content}\n\n"
            return formatted_str

        def format_for_retriever(value: dict) -> str:
            return value["input"]

        def format_for_prompt_template(value):
            # {input, context, history}
            new_value = {}
            new_value["input"] = value["input"]["input"]
            new_value["context"] = value["context"]
            new_value["history"] = value["input"]["history"]
            return new_value

        chain = (
            {
                "input": RunnablePassthrough(),
                "context": RunnableLambda(format_for_retriever) | retriever | format_document
            } | RunnableLambda(format_for_prompt_template) | self.prompt_template | print_prompt | self.chat_model | StrOutputParser()
        )

        conversation_chain = RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain


if __name__ == '__main__':
    # session id 配置
    session_config = {
        "configurable": {
            "session_id": "user_001",
        }
    }

    res = RagService().chain.invoke({"input": "SQL面试的题给我列举出一道"}, session_config)
    print(res)