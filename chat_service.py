from abc import ABC, abstractmethod

class ChatService(ABC):
    @abstractmethod
    def get_response(self, query: str) -> str:
        pass

    @abstractmethod
    def reload(self, new_model_path: str, new_model_type: str = None):
        pass
