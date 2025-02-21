import os

class Config:
    JSON_PATH = os.path.join("train", "data", "yourragfile.json")
    MODEL_PATH = r"C:\models\qtz8-vicuna-7b-v1.5.gguf" # quantized model with llama.cpp
    CHROMA_DIR = "chroma_db" # or FAISS
    RAG_KEYWORDS = ["rag_trigger"]
    FREE_PROMPT_TEMPLATE = (
        "YOUR PROMPT\n"
        "User: {query}\nCat:"
    )
