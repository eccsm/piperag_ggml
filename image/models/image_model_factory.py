from config import Config
from image_recognition_model import DeepfakeDetectionModel, ObjectRecognitionModel, ImageRecognitionModel


# Factory for image recognition models
class ImageModelFactory:
    _models = {
        "deepfake_detection": DeepfakeDetectionModel,
        "object_recognition": ObjectRecognitionModel,
    }

    @classmethod
    def get_model(cls, config: Config, task: str) -> ImageRecognitionModel:
        if task not in cls._models:
            raise ValueError(f"Unsupported task: {task}")
        return cls._models[task](config)


# Service class with improved separation of concerns
class ImageRecognitionService:
    def __init__(self, config: Config):
        self.config = config
        self._model_cache = {}  # Cache models to avoid reloading

    def get_model(self, task: str) -> ImageRecognitionModel:
        if task not in self._model_cache:
            self._model_cache[task] = ImageModelFactory.get_model(self.config, task)
        return self._model_cache[task]

    def recognize(self, image_bytes: bytes, task: str) -> str:
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        model = self.get_model(task)
        result = model.predict(image)

        # Format result as human-readable string
        score_percentage = result["score"] * 100
        if score_percentage < 50:
            interpretation = f"Uncertain: Possibly {result['label']}."
        elif score_percentage < 80:
            interpretation = f"Most likely {result['label']}."
        else:
            interpretation = f"Definitely {result['label']}."

        return f"{interpretation} (score: {score_percentage:.0f}%)"