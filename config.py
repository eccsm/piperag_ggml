# config.py
import os
from typing import Dict, List, Optional, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """
    Application configuration using Pydantic for validation.
    
    Environment variables override these settings.
    Example: export MODEL_TYPE=mlc_llm
    """
    
    # API Security
    API_KEY: str = Field(
        default="your-temp-dev-key-please-set-in-env",
        description="API key for authentication"
    )
    
    # File paths with env variable defaults
    JSON_PATH: str = Field(
        default=os.path.join("train", "data", "Ekincan_Casim_Chunks.json"),
        description="Path to training data JSON file"
    )
    GGUF_CACHE_DIR: str = Field(
        default="./vicuna-7b-q8",
        description="Directory for GGUF model cache"
    )
    CHROMA_DIR: str = Field(
        default="./chroma_db",
        description="Directory for Chroma vector DB"
    )
    
    # Model configuration
    GGUF_MODEL_REPO: str = Field(
        default="eccsm/vicuna-7b-q8",
        description="Repository for GGUF model"
    )
    GGUF_MODEL: str = Field(
        default="qtz8-vicuna-7b-v1.5.gguf",
        description="GGUF model filename"
    )
    MODEL_TYPE: str = Field(
        default="vicuna_ggml",
        description="Model type (vicuna_ggml or mlc_llm)"
    )
    DEFAULT_BRANCH: str = Field(
        default="main",
        description="Default git branch for model repos"
    )
    MLC_MODEL: str = Field(
        default="HF://eccsm/mlc_llm",
        description="MLC model path"
    )
    
    # RAG configuration
    RAG_KEYWORDS: List[str] = Field(
        default=["ekincan", "casim", "sug", "pep", "ecem"],
        description="Keywords that trigger RAG context inclusion"
    )
    FREE_PROMPT_TEMPLATE: str = Field(
        default="You are a funny cat named {cat_name}. Engage in a casual conversation.\nUser: {query}\nCat:",
        description="Template for non-RAG prompts"
    )
    
    # Image models configuration
    IMAGE_MODEL_MAP: Dict[str, str] = Field(
        default={
            "deepfake_detection": "prithivMLmods/Deepfake-Detection-Exp-02-22",
            "object_recognition": "Ultralytics/YOLOv8",
            "image_classification": "microsoft/resnet-50",
        },
        description="Mapping of task names to image model paths"
    )
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Model used for text embeddings"
    )
    
    # Derived properties
    @property
    def default_image_model(self) -> str:
        """Get the default image model from the map"""
        return self.IMAGE_MODEL_MAP["deepfake_detection"]
    
    # Validators
    @field_validator('MODEL_TYPE')
    def validate_model_type(cls, v):
        if v not in ["vicuna_ggml", "mlc_llm"]:
            raise ValueError(f"MODEL_TYPE must be either 'vicuna_ggml' or 'mlc_llm', got {v}")
        return v
    
    class Config:
        """Pydantic config"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True