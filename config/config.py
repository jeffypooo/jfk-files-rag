import os
import dotenv

dotenv.load_dotenv()

class Config:
    DEBUG = os.getenv("DEBUG") == "1"
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME")
    ALWAYS_RECREATE_COLLECTION = os.getenv("ALWAYS_RECREATE_COLLECTION") == "1"
    CHUNK_SIZE_WORDS = int(os.getenv("CHUNK_SIZE_WORDS"))
    OVERLAP_SIZE_WORDS = int(os.getenv("OVERLAP_SIZE_WORDS"))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")