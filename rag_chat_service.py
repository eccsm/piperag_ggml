import datetime
from chat_service import ChatService
from config import Config
from qa_service import QAChainBuilder


def get_cat_personality() -> str:
    day = datetime.datetime.today().day
    return "Sug" if day % 2 == 0 else "Pep"


class RAGChatService(ChatService):
    def __init__(self, config: Config):
        self.config = config
        self.qa_chain_builder = QAChainBuilder(config)

    def get_response(self, query: str) -> str:
        query_lower = query.lower()
        # Use RAG if any keyword matches.
        if any(keyword in query_lower for keyword in self.config.RAG_KEYWORDS):
            result = self.qa_chain_builder.chain({"query": query})
            return result.get("result", "")
        else:
            cat_name = get_cat_personality()
            prompt = self.config.FREE_PROMPT_TEMPLATE.format(cat_name=cat_name, query=query)
            result = self.qa_chain_builder.llm.invoke(prompt, max_tokens=256)
            if isinstance(result, dict) and "choices" in result:
                full_answer = result["choices"][0]["text"].strip()
            else:
                full_answer = str(result).strip()
            # Clean up echoes from the prompt.
            if full_answer.startswith(prompt):
                full_answer = full_answer[len(prompt):].strip()
            if "User:" in full_answer:
                full_answer = full_answer.split("User:")[0].strip()
            return full_answer

    def reload(self, new_model_path: str, new_model_type: str = None):
        self.qa_chain_builder.reload_model(new_model_path, new_model_type)
