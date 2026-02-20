from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

from langchain.embeddings import CacheBackedEmbeddings

# invoke my model in google
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", 
    temperature=0,
    max_output_tokens=2048,
    streaming=True  
)