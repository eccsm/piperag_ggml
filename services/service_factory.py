from typing import Dict, Type


from services.basic_chat_service import BasicChatService
from services.rag_chat_service import RAGChatService
from config import Config
from services.chat_service import ChatService


class ServiceFactory:
    # Registry of service implementations
    _services: Dict[str, Type[ChatService]] = {
        "vicuna_ggml": RAGChatService,
        "mlc_llm": BasicChatService,
    }

    @classmethod
    def register_service(cls, model_type: str, service_class: Type[ChatService]):
        """Register a new service implementation"""
        cls._services[model_type] = service_class

    @classmethod
    def create_service(cls, config: Config) -> ChatService:
        """Create appropriate service based on configuration"""
        if config.MODEL_TYPE not in cls._services:
            raise ValueError(f"Unsupported MODEL_TYPE: {config.MODEL_TYPE}")

        service_class = cls._services[config.MODEL_TYPE]
        return service_class(config)