from langgraph.graph import StateGraph, END
from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import ollama

# ---- Setup (same as before) ----
loader = PyPDFLoader("data.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.from_documents(chunks, embeddings)

# ---- State ----
class State(TypedDict):
    query: str
    context: str
    answer: str

# ---- Nodes ----

def retrieve(state: State):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["query"])
    context = " ".join([doc.page_content for doc in docs])
    return {"context": context}

def generate(state: State):
    prompt = f"""
Answer ONLY from context.

Context:
{state['context']}

Question:
{state['query']}

Explain clearly:
"""
    response = ollama.chat(
        model="phi",
        messages=[{"role": "user", "content": prompt}]
    )
    return {"answer": response["message"]["content"]}

# ---- Graph ----
builder = StateGraph(State)

builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()

# ---- Run ----
query = input("Ask something: ")

result = graph.invoke({"query": query})

print("\nAnswer:\n")
print(result["answer"])
