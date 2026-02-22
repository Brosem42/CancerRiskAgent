from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import os
import pathlib as Path
import pathlib
from dotenv import load_dotenv
#add sha256 encoder to mimic live setting where data privacy is key--with proprietary PHI data

load_dotenv()

#create caching for my embeddings to lower costs 
cache_dir = pathlib.Path("cache_uploads/embedding_cache_folder")
cache_dir.mkdir(parents=True, exist_ok=True)
store = LocalFileStore(str(cache_dir))

underlying_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
#prevent unecessary costs by caching my emdbeddings
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model 
)
