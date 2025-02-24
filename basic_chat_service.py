import logging

from posthog import flush

from mlc_llm import MLCEngine

from chat_service import ChatService  # Your abstract chat service interface
from config import Config

logger = logging.getLogger("mlc_llm_integration")
logger.setLevel(logging.DEBUG)
stop_tokens = ["User:"]
class BasicChatService(ChatService):
    def __init__(self, config: Config):
        self.config = config
        # Instantiate the MLC wrapper with the HF URL from config.
        self.llm = MLCEngine(self.config.MLC_MODEL)

    def get_response(self, query: str) -> str:
        full_response = ""
        messages = [{"role": "user", "content": query}]
        try:
            for response in self.llm.chat.completions.create(messages=messages, stream=True):
                for choice in response.choices:
                    content = choice.delta.content.rstrip()
                    full_response += content
                    logger.debug(f"MLC response chunk: {content}")
            return full_response
        except Exception as e:
            logger.error("Error in MLC _call", exc_info=True)
            raise e


    def reload(self, new_model_path: str, new_model_type: str = None):
        # Update config and reinitialize the wrapper if needed.
        if new_model_type:
            self.config.MODEL_TYPE = new_model_type
        if new_model_path:
            self.config.MLC_MODEL = new_model_path
        self.llm = MLCEngine(self.config.MLC_MODEL)
