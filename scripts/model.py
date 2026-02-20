from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()

from langchain.embeddings import CacheBackedEmbeddings
api_key = os.getenv("GOOGLE_API_KEY")

# invoke my model in google
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    GOOGLE_API_KEY=api_key,
    temperature=0,
    max_output_tokens=2048,
    streaming=True  
)