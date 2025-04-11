from dotenv import load_dotenv
load_dotenv()
import chromadb
import os

from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PandasCSVReader
from llama_cloud_services import LlamaParse

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.cohere import CohereEmbedding

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import load_index_from_storage

from llama_index.llms.anthropic import Anthropic
from llama_index.core.memory import ChatMemoryBuffer

USE_PERSISTENCE = True  
CHROMA_PATH = './chroma_db'
COLLECTION_NAME = 'investments'
INDEX_PERSIST_DIR = './storage'
FORCE_RECREATE_INDEX = False
TOKENIZERS_PARALLELISM = False

class Assistant:
    def __init__(self):
        self.documents = None
        self.embed_model = None
        self.embed_model_query = None
        self.chroma_collection = None
        self.storage_context = None
        self.vector_store = None
        
    def load_data(self):
        csv_reader = PandasCSVReader(concat_rows=False)
        file_extractor = {".csv": csv_reader}
        documents = SimpleDirectoryReader(input_dir="../data", file_extractor=file_extractor, recursive=True).load_data()
        print(f"Total of {len(documents)} documents loaded.")
        
        self.documents = documents
        
    def initialize_embeddings(self):
        # self.embed_model = HuggingFaceEmbedding(model_name='intfloat/multilingual-e5-large')

        self.embed_model = CohereEmbedding(
            api_key=os.environ.get("COHERE_API_KEY"),
            model_name="embed-multilingual-v3.0",
            input_type="search_document",
        )

        self.embed_model_query = CohereEmbedding(
            api_key=os.environ.get("COHERE_API_KEY"),
            model_name="embed-multilingual-v3.0",
            input_type="search_query",
        )
        
    def chroma_config(self):
        # ChromaDB configuration
        if USE_PERSISTENCE:
            
            if not os.path.exists(CHROMA_PATH):
                os.makedirs(CHROMA_PATH)
            chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        else:
            chroma_client = chromadb.EphemeralClient()

        # Wrapper compatibility with Chroma
        class ChromaEmbeddingWrapper:
            def __init__(self, embed_model):
                self.model = embed_model
            
            def __call__(self, input):
                return self.model.embed(input)

        embed_wrapper = ChromaEmbeddingWrapper(self.embed_model)

        try:
            self.chroma_collection = chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=embed_wrapper
            )
            print(f"Collection '{COLLECTION_NAME}' loaded successfully")
        except Exception as e:
            print(f'Error with the collection: {e}')
            raise
        
    def create_index(self):
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # with persistence (without recreating the index)
        if USE_PERSISTENCE and os.path.exists(INDEX_PERSIST_DIR) and not FORCE_RECREATE_INDEX:
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=INDEX_PERSIST_DIR)
            index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        else:
            index = VectorStoreIndex(self.documents, storage_context=storage_context, embed_model=self.embed_model)

            if USE_PERSISTENCE:
                if not os.path.exists(INDEX_PERSIST_DIR):
                    os.makedirs(INDEX_PERSIST_DIR)
                
                index.storage_context.persist(persist_dir=INDEX_PERSIST_DIR)
                print(f"Index persisted in {INDEX_PERSIST_DIR}")
                
        return index
    
    
    def get_chroma_collection(self):
        return self.chroma_collection
    
    def get_embedding(self):
        return self.embed_model_query
    
    def llm(self):
        self.load_data()
        self.initialize_embeddings()
        self.chroma_config() 
        
        index = self.create_index()
        
        llm = Anthropic(model="claude-3-5-sonnet-latest", temperature=0.5, max_tokens=1024, timeout=None, max_retries=2)
        memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

        chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            llm=llm,
            chat_memory=memory,
            embed_model=self.embed_model_query,
            similarity_top_k=40,
            verbose=True,  
            system_prompt=("""
                Sua persona é um assistente pessoal de análise de investimentos. Seu trabalho é responder perguntas relacionadas aos investimentos do usuário
                dado o contexto. Os dados contem informações sobre dividendos de cada mês e sobre as aplicações feitas em cada mês.              
            """)
        )
        
        return chat_engine
    
        
        



