import os
import torch
from huggingface_hub import hf_hub_download
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_huggingface import HuggingFaceEmbeddings

from data_loader import DataLoader
from config import Config


class MultiChainBuilder:
    """
    Builds and stores multiple QA chains for different data files:
      - ekincan_tr and ekincan_en
      - pep_tr and pep_en
      - sug_tr and sug_en

    Also builds a fallback LLM instance for free-form conversation.
    """
    def __init__(self, config: Config):
        self.config = config
        self.chains = {}

        # Build separate chains for each combination
        self._build_ekincan_chain_tr()
        self._build_ekincan_chain_en()
        self._build_pep_chain_tr()
        self._build_pep_chain_en()
        self._build_sug_chain_tr()
        self._build_sug_chain_en()

        # Build a fallback LLM for free-form conversation
        self.llm = self._build_llm()

    def _build_chain_for_file(self, json_path: str) -> RetrievalQA:
        """
        Helper to build a single chain from a given JSON file path:
          1) Load documents via data_loader
          2) Create embeddings + Chroma
          3) Initialize a LlamaCpp LLM
          4) Create a RetrievalQA chain
        """
        # 1) Load chunked docs
        loader = DataLoader(json_path)
        docs = loader.load_documents()

        # 2) Create embeddings + vector store
        embeddings = HuggingFaceEmbeddings(model_name=self.config.EMBEDDING_MODEL)
        chroma_dir = self.config.CHROMA_DIR

        if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
            vectordb = Chroma(
                embedding_function=embeddings,
                persist_directory=chroma_dir
            )
        else:
            vectordb = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=chroma_dir
            )
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # 3) LLM loading (used only within this chain)
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

        # 4) Build and return the QA chain
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )

    def _build_llm(self):
        """
        Builds a fallback LLM instance for free-form conversation.
        """
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
        return llm

    def _build_ekincan_chain_tr(self):
        if hasattr(self.config, "EKINCAN_JSON_PATH_TR"):
            json_path = self.config.EKINCAN_JSON_PATH_TR
            if json_path:
                chain = self._build_chain_for_file(json_path)
                self.chains["ekincan_tr"] = chain

    def _build_ekincan_chain_en(self):
        if hasattr(self.config, "EKINCAN_JSON_PATH_EN"):
            json_path = self.config.EKINCAN_JSON_PATH_EN
            if json_path:
                chain = self._build_chain_for_file(json_path)
                self.chains["ekincan_en"] = chain

    def _build_pep_chain_tr(self):
        if hasattr(self.config, "PEP_JSON_PATH_TR"):
            json_path = self.config.PEP_JSON_PATH_TR
            if json_path:
                chain = self._build_chain_for_file(json_path)
                self.chains["pep_tr"] = chain

    def _build_pep_chain_en(self):
        if hasattr(self.config, "PEP_JSON_PATH_EN"):
            json_path = self.config.PEP_JSON_PATH_EN
            if json_path:
                chain = self._build_chain_for_file(json_path)
                self.chains["pep_en"] = chain

    def _build_sug_chain_tr(self):
        if hasattr(self.config, "SUG_JSON_PATH_TR"):
            json_path = self.config.SUG_JSON_PATH_TR
            if json_path:
                chain = self._build_chain_for_file(json_path)
                self.chains["sug_tr"] = chain

    def _build_sug_chain_en(self):
        if hasattr(self.config, "SUG_JSON_PATH_EN"):
            json_path = self.config.SUG_JSON_PATH_EN
            if json_path:
                chain = self._build_chain_for_file(json_path)
                self.chains["sug_en"] = chain

    def get_chain(self, topic: str) -> RetrievalQA:
        return self.chains.get(topic)
