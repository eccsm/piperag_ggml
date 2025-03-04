from typing import Dict, Type
import logging

from services.basic_chat_service import BasicChatService
from services.rag_chat_service import RAGChatService
from config import Config
from services.chat_service import ChatService

logger = logging.getLogger(__name__)


class ServiceFactory:
    _services: Dict[str, Type[ChatService]] = {
        "vicuna_ggml": RAGChatService,
        "mlc_llm": BasicChatService,
    }

    @classmethod
    def register_service(cls, model_type: str, service_class: Type[ChatService]):
        """
        Register a new service implementation

        Args:
            model_type: The model type key to register
            service_class: The service class to associate with the model type
        """
        logger.info(f"Registering service for model type: {model_type}")
        cls._services[model_type] = service_class

    @classmethod
    def create_service(cls, config: Config) -> ChatService:
        """
        Create appropriate service based on configuration

        Args:
            config: Configuration object

        Returns:
            Instantiated chat service

        Raises:
            ValueError: If model type is not supported
        """
        logger.info(f"Attempting to create service for model type: {config.MODEL_TYPE}")

        matching_services = [
            service_class for model_type, service_class in cls._services.items()
            if config.MODEL_TYPE.startswith(model_type)
        ]

        if not matching_services:
            supported_types = list(cls._services.keys())
            logger.error(f"Unsupported MODEL_TYPE: {config.MODEL_TYPE}")
            raise ValueError(
                f"Unsupported MODEL_TYPE: {config.MODEL_TYPE}. "
                f"Supported types: {supported_types}"
            )

        service_class = matching_services[0]

        try:
            service_instance = service_class(config)
            logger.info(f"Successfully created service: {service_class.__name__}")
            return service_instance
        except Exception as e:
            logger.error(f"Failed to instantiate service: {e}", exc_info=True)
            raise