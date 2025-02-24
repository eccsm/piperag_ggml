# config.py
import os

class Config:
    # For text queries with RAG support (e.g., Vicuna GGML)
    JSON_PATH = os.path.join("train", "data", "Ekincan_Casim_Chunks.json")
    GGUF_MODEL_REPO = "eccsm/vicuna-7b-q8"
    GGUF_CACHE_DIR = "./vicuna-7b-q8"
    GGUF_MODEL = "qtz8-vicuna-7b-v1.5.gguf"
    CHROMA_DIR = "chroma_db"
    RAG_KEYWORDS = ["ekincan", "casim", "sug", "pep", "ecem"]
    FREE_PROMPT_TEMPLATE = (
        "You are a funny cat named {cat_name}. Engage in a casual conversation.\n"
        "User: {query}\nCat:"
    )

    MODEL_TYPE="vicuna_ggml"
    DEFAULT_BRANCH="main"
    MLC_MODEL = "HF://eccsm/mlc_llm"


    # For image-based models:
    IMAGE_MODEL_MAP = {
        "deepfake_detection": "prithivMLmods/Deepfake-Detection-Exp-02-22",
        "object_recognition": "Ultralytics/YOLOv8",
        "image_classification": "microsoft/resnet-50",
    }
    DEFAULT_IMAGE_MODEL = IMAGE_MODEL_MAP["deepfake_detection"]
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
