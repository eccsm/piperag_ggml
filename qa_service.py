import os
import torch  # For GPU check
import datetime

from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

from config import Config
from data_loader import DataLoader

class QAChainBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(self.config.JSON_PATH)
        self.chain, self.llm = self._create_chain()

    def _create_chain(self):
        # Load documents and create embeddings.
        docs = self.data_loader.load_documents()
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Build or load the Chroma vector store.
        if os.path.exists(self.config.CHROMA_DIR) and os.listdir(self.config.CHROMA_DIR):
            vectordb = Chroma(
                embedding_function=embeddings,
                persist_directory=self.config.CHROMA_DIR
            )
        else:
            vectordb = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=self.config.CHROMA_DIR
            )

        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Set up callback for streaming.
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Check GPU availability and set GPU layers accordingly.
        use_cuda = torch.cuda.is_available()
        n_gpu_layers = 30 if use_cuda else 0

        # Initialize the local Llama model.
        llm = LlamaCpp(
            model_path=self.config.MODEL_PATH,
            n_ctx=2048,
            temperature=0.7,
            max_tokens=512,
            top_p=0.95,
            callback_manager=callback_manager,
            verbose=True,
            stop=["User:"],
            n_gpu_layers=n_gpu_layers
        )

        # Build the RetrievalQA chain.
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False
        )
        return qa_chain, llm


class CatPersonality:
    @staticmethod
    def get_personality() -> str:
        """
        Returns a cat personality name based on the day of the month.
        'Sug' on even days, 'Pep' on odd days.
        """
        day = datetime.datetime.today().day
        return "Sug" if day % 2 == 0 else "Pep"
