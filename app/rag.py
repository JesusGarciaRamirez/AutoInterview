from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from autointerview.agents.selfRAG import selfRAG
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import pprint

class ChatReflective:
    model = None
    retriever = None
    def __init__(self, candidate, llm_model="mistral:instruct", profiles_dir="Profiles"):
        self.model = selfRAG().get_agent()
        self.candidate = candidate
        self.llm_model = llm_model
        self.profiles_dir = profiles_dir

    def create_index(self):
        loader = PyPDFDirectoryLoader(os.path.join(self.profiles_dir, self.candidate))
        docs = loader.load()
        # Split
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, 
            chunk_overlap=100)

        # Make splits
        splits = text_splitter.split_documents(docs)

        # Initialize embeddings (HuggingFace)
        modelPath = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,    
            model_kwargs=model_kwargs, 
            encode_kwargs=encode_kwargs 
        )

        # Index
        vectorstore = Chroma.from_documents(
            documents=splits,
            persist_directory="db/chroma/RAG",
            collection_name=self.candidate,
            embedding=embeddings,
        )
        self.retriever = vectorstore.as_retriever()

    def ask(self, query: str):
        if not self.model:
            return "Model not initialized correctly"
        # Run agent
        inputs = {"keys": {"question": query, "model": self.llm_model, "retriever": self.retriever}}
        for output in self.model.stream(inputs):
            for key, value in output.items():
                # Node
                pprint.pprint(f"Node '{key}':")
                # Optional: print full state at each node
                # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            pprint.pprint("\n---\n")

        return value['keys']['generation']
    
    def clear(self):
        self.model = None
        self.retriever = None

