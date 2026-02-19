from typing import Annotated
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from typing_extensions import List, TypedDict
from langgraph.graph.message import add_messages
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts import DocumentBaseRetriever, EMBEDDINGS, llm, load_document


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

# invoke a retriever and a prompt
retriever = DocumentBaseRetriever()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)

# define the state of out graph
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    issues_report: str
    issues_detected: str
    messages: Annotated[list, add_messages]

# defining my 4 nodes for retrieval in langgraph for retrieve, generate, double_check, and doc_finalizer

#start with retrieval
def retrieve(state: State):
    retrieved_docs = retriever.invoke(state["messages"][-1].content)
    print(retrieved_docs)
    return {"context": retrieved_docs}

#generate function
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke(
        {"question": state["messages"][-1].content, "content": docs_content}
    )
    response = llm.invoke(messages)
    print(response.content)
    return {"answer": response.content}

#add validation content check
#in production I would implement human in the loop techniques as a verification process, 
# here in this instance I will just use an llm to implement verfication of content issues
# implment validation with a thinking step
def double_check(state: State):
    result = llm.invoke([{
        "role": "user",
        "content": (
            f"Review the following clinical decision support project for any violations of HIPAA compliance rules for quality assurance, PHI/PII breach of security, and clinical delivery standards that align with a high level of patient care."
            f"Return 'ISSUES FOUND' followed by any issues detected or 'NO ISSUES': {state['answer']}"
        )
    }]) 
    #extract actual response after thinking block
    content = result.content
    if "</think>" in content:
        actual_response = content.split("</think>", 1)[1].strip()
    else:
        actual_response = content.strip()

    if "ISSUES FOUND" in actual_response:
        print("issues detected")
        return {
            "issues_report": actual_response.split("ISSUES FOUND", 1)[1].strip(),
            "issues_detected": True
        }
    print("no issues detected")
    return {
        "issues_report": "",
        "issues_detected": False
    }


# final node to integrate feedback to produce finalized, compliant docs
#feedback looping functions to mimic feedback loop for patient order placement or clinical decision workflow steps for patient care
def doc_finalizer(state: State):
    """
    Finalize patient user query by integrating feedback.
    """
    if "issues_detected" in state and state["issues_detected"]:
        response = llm.invoke(
            messages=[{
                "role": "user",
                "content": (
                    f"Revise the following patient document to address these feedback points:{state['issues_report']}\n"
                    f"Original Document: {state['answer']}\n"
                    f"Always return the full revised document, even if no changes are needed."

                )
            }]
        )
        return {
            "messages": [AIMessage(response.content)]
        }
    return {
        "messages": [AIMessage(state["answer"])]
    }

# build our knowledge graph to passs to agent
graph_builder = StateGraph(State).add_sequence(
    [retrieve, generate, double_check, doc_finalizer])
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}


