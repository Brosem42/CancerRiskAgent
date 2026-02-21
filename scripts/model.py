from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

from langchain.embeddings import CacheBackedEmbeddings
# Get the key (Works on Mac via .env and on Render via Dashboard)
api_key = os.getenv("GOOGLE_API_KEY")

# invoke my model in google
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0,
    max_output_tokens=2048,
    streaming=False  
)