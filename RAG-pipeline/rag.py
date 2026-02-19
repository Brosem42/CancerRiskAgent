from typing import Annotated
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from langgraph.constants import END
from typing_extensions import List, TypedDict

#module imports
from scripts.doc_retrieval import DocumentBaseRetriever
from scripts.model import llm

# define our system_prompt
system_prompt = (
    " You are a helpful clinical AI assistant who is tasked with retrieving the correct patient data. "
    " This will be based on a given user input query for a specific patient in the patients.json file. "
    " Your task is to retrieve the most relevant information to serve as a similarity search lookup for allied health personnel at a maximum similarity score. "
    " Do not make up responses. Ensure that every answer or response includes a citation of where you retrieved that data in reference to the patients.json."
    " If none of the data from patients.json is relevant to the user query, return a response that states "
    " there is no relevant data, and then answer the question to the best of your expert knowledge of the field."
    "\n\nHere is the patient data:"
    "{context}"
)