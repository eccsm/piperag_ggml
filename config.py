import os

class Config:
    JSON_PATH = os.path.join("train", "data", "Ekincan_Casim_Chunks.json")
    MODEL_PATH = r"C:\models\qtz8-vicuna-7b-v1.5.gguf"
    CHROMA_DIR = "chroma_db"
    RAG_KEYWORDS = ["ekincan", "casim","sug","pep","ecem"]
    FREE_PROMPT_TEMPLATE = (
        "You are a funny cat named {cat_name}. Engage in a casual conversation.\n"
        "User: {query}\nCat:"
    )
