# load dotenv + imports for retriever tool
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain import hub

# Load environment variables from .env file
load_dotenv()