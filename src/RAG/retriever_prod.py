# imports 
# imports 
# # load dotenv + imports for retriever tool
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma 
from langchain.tools import tool
import langchainhub as hub
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import vertexai
from vertexai.evaluation import EvalTask
from vertexai.language_models import TextEmbeddingModel
from langchain_core.runnables import Runnable

#helper import
from RAG.citations import format_sources_with_citations, infer_page_from_text

import os
from dotenv import load_dotenv
# Load environment variables from .env file first
load_dotenv()

# load and set environments---safe because no actual var are exposed, just using a flag method
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
GOOGLE_GENAI_USE_VERTEXAI = os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
GOOGLE_CLOUD_PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"] = "gen-lang-client-0343643614"
GCLOUD_PROJECT = os.environ["GCLOUD_PROJECT"] = "gen-lang-client-0343643614"
GOOGLE_CLOUD_LOCATION = os.environ["GOOGLE_CLOUD_LOCATION"] = "us-east4"
GOOGLE_APPLICATION_CREDENTIALS = os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/briannamitchell/.config/gcp/vertex-sa.json"


# Load environment variables from chromaDB with persistence locally, and prevent unecessary calls to API--more cost-effective approach
persistent_client = chromadb.PersistentClient(path="./chroma_db")

# Importing Gemini for embedding
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# adding response enhancement for generated outputs
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

#vectore store
vectorstore = Chroma(
    collection_name="ng12",
    embedding_function=embeddings,
    client=persistent_client
)

#context
documents = [
    Document(page_content="Shortness of breath with cough or fatigue or chest pain or weight loss or appetite loss (unexplained), 40 and over: possible cancer Lung or mesothelioma", metadata={"referral": "Urgent", "source": "Suspected cancer: recognition and referral (NG12) 2026", "page": "52"}),
    Document(page_content="Bleeding, bruising or petechiae, unexplained: possible cancer Leukaemia", metadata={"referral": "Very urgent", "source": "Suspected cancer: recognition and referral (NG12) 2026", "page": "43"}),
    Document(page_content="Fracture unexplained, 60 and over: possible cancer Myeloma", metadata={"referral": "Unexplained", "source": "Suspected cancer: recognition and referral (NG12) 2026", "page": "55"}),
    Document(page_content="Refer people using a suspected cancer pathway referral for oesophageal cancer if they: have dysphagia or, are aged 55 and over, with weight loss, and they have any of the following: upper abdominal pain, reflux, dyspepsia. [2015, amended 2025]", metadata={"referral": "Suspected cancer pathway referral", "source": "Suspected cancer: recognition and referral (NG12) 2026", "page": "11"}),
    Document(page_content="Skin lesion that raises the suspicion of a basal cell carcinoma: possible cancer Basal cell carcinoma  ", metadata={"referral": "Raises the suspicion of", "source": "Suspected cancer: recognition and referral (NG12) 2026", "page": "58"}),
    Document(page_content="Urinary urgency or frequency, increased and persistent or frequent, particularly more than 12 times per month in women, especially if 50 and over: possible cancer Ovarian", metadata={"referral": "Persistent", "source": "Suspected cancer: recognition and referral (NG12) 2026", "page": "60"}),
    Document(page_content="Upper abdominal pain with low haemoglobin levels or raised platelet count or nausea or vomiting, 55 and over: possible cancer Oesophageal or stomach ", metadata={"referral": "Non-urgent", "source": "Suspected cancer: recognition and referral (NG12) 2026", "page": "40"}),
    Document(page_content="Petechiae unexplained in children and young people: possible cancer Leukaemia", metadata={"referral": "Immediate", "source": "Suspected cancer: recognition and referral (NG12) 2026", "page": "74"})
]

#create retriever from vector store and then apply max marginal relevance
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,
        "fetch_k": 50,
        "lambda_mult": 0.6
    }
)

# attrbution
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

attribution_prompt = ChatPromptTemplate.from_template(
    """ You are a precise and accurate clinical AI assistant that provides expert knowledge in
    clinical decision-making support for those who encounter direct patient care.
    You have inherited a persona profile module as your agentic architecture.
    Main Task: Your main task is to determine whether the presented patient requires an urgent referral or not.
    Use the information provided from the text corpus below:
    The National Institute for Health and Care Excellence (NICE) Guideline for Suspected cancer: recognition and referral NICE guideline.
    Once, main task is complete and accurate, you must provide a recommendation that decides post-referral instructions corresponding to the most relevant medical imaging practices.
    If patient does not meet urgent referral criteria, do not recommend any medical imaging practices.

    Answer the following question based ONLY on the provided sources. 
    For each fact or claim in your answer include a citation that refers to the source.

    Do not make up information or provide personal opinions in your responses without verifying answers with evidence.
    You must cite the specific sources you found from the NICE guidelines in this specific format below:
    This is how you are expected to format: [referral type: insert referral, source: name of text corpus, (year published), page: insert page number that you find the passage from].
    If page number is not available, use the helper function _PAGE_RE to calculate page number.
    
    Your source attributes at the end of your responses will look like this template below in practice:
    [referral: Persistent, source: Suspected cancer: recognition and referral (NG12), 2026, page: 60]

    How your input and output will be formatted with citation sources used:
    Question: {question}

    Sources: {sources}

    Your answer:
    """
)

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)

def generated_attributed_response(question: str):
    retrieved_docs = retriever.invoke(question)
    sources_formatted  = format_sources_with_citations(retrieved_docs)
    attribution_chain = attribution_prompt | llm | StrOutputParser()

    response = attribution_chain.invoke({
        "question": question,
        "sources": sources_formatted
    })
    return response

question = "Based on this patient data provided, what type of referral is required for patient care?--data:'{\"patient_id\": \"PT-103\", \"name\": \"Robert Brown\", \"age\": \"45\", \"gender\": \"Male\", \"smoking_history\": \"Ex-Smoker\", \"symptoms\": [\"persistent cough\", \"shortness of breath\"], \"symptom_duration_days\": 28}'"
response = generated_attributed_response(question)
print(response)




