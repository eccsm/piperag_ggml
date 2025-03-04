from services.chat_service import ChatService
from langdetect import detect, LangDetectException
from services.multi_chain_builder import MultiChainBuilder

def _is_turkish(text: str) -> bool:
    try:
        return detect(text) == "tr"
    except LangDetectException:
        return False

class RAGChatService(ChatService):
    def __init__(self, config):
        super().__init__(config)
        self.chain_builder = MultiChainBuilder(config)

    def generate_response(self, query: str, **kwargs) -> str:
        is_turkish = _is_turkish(query)
        query_lower = query.lower()

        if "ekincan" in query_lower:
            chain_key = "ekincan_tr" if is_turkish else "ekincan_en"
            chain = self.chain_builder.get_chain(chain_key)
            if chain:
                result = chain({"query": query})
                return result.get("result", "")
            else:
                return f"No chain for {chain_key}"

        if "pep" in query_lower:
            chain_key = "pep_tr" if is_turkish else "pep_en"
            chain = self.chain_builder.get_chain(chain_key)
            if chain:
                result = chain({"query": query})
                return result.get("result", "")
            else:
                return f"No chain for {chain_key}"

        if "sug" in query_lower:
            chain_key = "sug_tr" if is_turkish else "sug_en"
            chain = self.chain_builder.get_chain(chain_key)
            if chain:
                result = chain({"query": query})
                return result.get("result", "")
            else:
                return f"No chain for {chain_key}"

        # Fallback: free-form conversation using the fallback LLM.
        cat_name = self.config.CAT_NAME_TR if is_turkish else self.config.CAT_NAME_EN
        prompt = self.config.FREE_PROMPT_TEMPLATE.format(cat_name=cat_name, query=query)

        raw_result = self.chain_builder.llm.generate([prompt], max_tokens=256)
        full_answer = raw_result.generations[0][0].text.strip()

        if full_answer.startswith(prompt):
            full_answer = full_answer[len(prompt):].strip()
        if "User:" in full_answer:
            full_answer = full_answer.split("User:")[0].strip()

        return full_answer

