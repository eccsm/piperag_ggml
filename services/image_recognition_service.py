import io
from PIL import Image
import torch
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from config import Config


class ImageRecognitionService:
    def __init__(self, config: Config, task: str = None):
        self.config = config
        self.task = task
        self.device = 0 if torch.cuda.is_available() else -1

        if task == "object_recognition":
            # Use the Ultralytics YOLO library for object detection.
            # Make sure to install ultralytics (pip install ultralytics) and have the YOLOv5n model file available.
            from ultralytics import YOLO
            # Adjust the model path if needed; here we assume a local file "yolov5n.pt"
            self.yolo = YOLO("yolov5n.pt")
        elif task == "image_classification":
            # Option 1: Use pipeline directly with the model name from config.
            model_name = self.config.IMAGE_MODEL_MAP.get(task, self.config.DEFAULT_IMAGE_MODEL)
            try:
                self.pipe = pipeline("image-classification", model=model_name, device=self.device)
            except Exception as e:
                # Optionally, if the pipeline fails, load manually:
                from transformers import AutoImageProcessor, AutoModelForImageClassification
                processor = AutoImageProcessor.from_pretrained(model_name)
                model = AutoModelForImageClassification.from_pretrained(model_name)
                self.pipe = pipeline("image-classification", model=model, tokenizer=processor, device=self.device)
        else:
            # Default: use deepfake_detection pipeline.
            model_name = self.config.IMAGE_MODEL_MAP.get(task, self.config.DEFAULT_IMAGE_MODEL)
            self.pipe = pipeline("image-classification", model=model_name, device=self.device)

    def recognize(self, image_bytes: bytes) -> str:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        if self.task == "object_recognition":
            # Use YOLO for object detection.
            results = self.yolo(image)
            if results and len(results) > 0:
                detection = results[0]
                if detection.boxes.cls.numel() > 0:
                    cls_id = int(detection.boxes.cls[0].item())
                    label = detection.names.get(cls_id, "Unknown")
                    score = detection.boxes.conf[0].item()
                    score_percentage = score * 100
                    if score_percentage < 50:
                        interpretation = f"Uncertain: Possibly {label}."
                    elif score_percentage < 80:
                        interpretation = f"Most likely {label}."
                    else:
                        interpretation = f"Definitely {label}."
                    return f"{interpretation} (score: {score_percentage:.0f}%)"
            return "No detection"
        else:
            # For image classification tasks.
            results = self.pipe(image)
            if results:
                label = results[0]["label"]
                score = results[0]["score"]
                score_percentage = score * 100  # Convert to percentage.
                if score_percentage < 50:
                    interpretation = f"Uncertain: Possibly {label}."
                elif score_percentage < 80:
                    interpretation = f"Most likely {label}."
                else:
                    interpretation = f"Definitely {label}."
                return f"{interpretation} (score: {score_percentage:.0f}%)"
            return "No result"

