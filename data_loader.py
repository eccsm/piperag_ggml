import json
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

class DataLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path
        self.splitter = CharacterTextSplitter(
            separator="\n\n",
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )

    def load_documents(self) -> list:
        """Loads the JSON file, splits texts into chunks, and returns a list of Document objects."""
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []
        for item in data:
            text = item["text"]
            chunks = self.splitter.split_text(text)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "id": item["id"],
                        "section": item["section"],
                        "title": item["title"],
                    }
                )
                docs.append(doc)
        return docs
