import os

import torch
from huggingface_hub import hf_hub_download
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings

from config import Config
from data_loader import DataLoader


class QAChainBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(self.config.JSON_PATH)
        self.chain, self.llm = self._create_chain()

    def _create_chain(self):
        print(f"Loading model: {self.config.MODEL_TYPE}")

        # Load documents and create embeddings.
        docs = self.data_loader.load_documents()
        embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)

        # Create or load a Chroma vector store.
        if os.path.exists(self.config.CHROMA_DIR) and os.listdir(self.config.CHROMA_DIR):
            vectordb = Chroma(embedding_function=embeddings, persist_directory=self.config.CHROMA_DIR)
        else:
            vectordb = Chroma.from_documents(
                documents=docs, embedding=embeddings, persist_directory=self.config.CHROMA_DIR
            )
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Dynamically load the model based on the model type.
        if self.config.MODEL_TYPE == "vicuna_ggml":
            callbacks = [StreamingStdOutCallbackHandler()]
            use_cuda = torch.cuda.is_available()
            n_gpu_layers = 30 if use_cuda else 0

            model_path = hf_hub_download(
                repo_id=self.config.GGUF_MODEL_REPO,
                filename=self.config.GGUF_MODEL,
                revision=self.config.DEFAULT_BRANCH,
                cache_dir=self.config.GGUF_CACHE_DIR
            )

            llm = LlamaCpp(
                model_path=model_path,
                n_ctx=2048,
                temperature=0.7,
                max_tokens=512,
                top_p=0.95,
                callback_manager=callbacks,
                verbose=True,
                stop=["User:"],
                n_gpu_layers=n_gpu_layers
            )
        elif self.config.MODEL_TYPE == "mlc_llm":
            from mlc_llm import MLCEngine
            llm = MLCEngine(self.config.MLC_MODEL)
        else:
            raise ValueError(f"Unsupported model type: {self.config.MODEL_TYPE}")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        return qa_chain, llm

    def reload_model(self, new_model_path: str = None, new_model_type: str = None):
        if new_model_type:
            self.config.MODEL_TYPE = new_model_type
        if new_model_path:
            self.config.MLC_MODEL = new_model_path
        print(f"Reloading model: {self.config.MODEL_TYPE} from {new_model_path or self.config.MLC_MODEL}")
        self.chain, self.llm = self._create_chain()
