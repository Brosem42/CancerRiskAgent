from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
import hashlib
import pathlib as Path
import pathlib
from dotenv import load_dotenv
#add sha256 encoder to mimic live setting where data privacy is key--with proprietary PHI data

load_dotenv()

#create caching for my embeddings to lower costs 
cache_dir = pathlib.Path.cwd() / "embedding_cache_folder"
store = LocalFileStore(cache_dir)

underlying_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

#prevent unecessary costs by caching my emdbeddings
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store
)
