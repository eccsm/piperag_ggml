from abc import ABC, abstractmethod
import io
from PIL import Image
import torch
from config import Config


class ImageRecognitionModel(ABC):
    @abstractmethod
    def predict(self, image) -> dict:
        """Predict from image and return results"""
        pass


class DeepfakeDetectionModel(ImageRecognitionModel):
    def __init__(self, config: Config):
        from transformers import pipeline
        model_name = config.IMAGE_MODEL_MAP["deepfake_detection"]
        self.pipe = pipeline("image-classification", model=model_name,
                             device=0 if torch.cuda.is_available() else -1)

    def predict(self, image) -> dict:
        results = self.pipe(image)
        if not results:
            return {"label": "Unknown", "score": 0.0}
        return {"label": results[0]["label"], "score": results[0]["score"]}


class ObjectRecognitionModel(ImageRecognitionModel):
    def __init__(self, config: Config):
        from ultralytics import YOLO
        self.yolo = YOLO("yolov5n.pt")

    def predict(self, image) -> dict:
        results = self.yolo(image)
        if not results or len(results) == 0:
            return {"label": "No detection", "score": 0.0}

        detection = results[0]
        if detection.boxes.cls.numel() == 0:
            return {"label": "No detection", "score": 0.0}

        cls_id = int(detection.boxes.cls[0].item())
        label = detection.names.get(cls_id, "Unknown")
        score = detection.boxes.conf[0].item()

        return {"label": label, "score": score}


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