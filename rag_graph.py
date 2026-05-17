from langgraph.graph import StateGraph, END
from typing import TypedDict
import ollama

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---- State ----
class State(TypedDict):
    query: str
    context: str
    answer: str
    sources: list


# ---- Build DB dynamically ----
def create_db(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 🚨 Check if PDF has text
    if not docs or all(len(d.page_content.strip()) == 0 for d in docs):
        raise ValueError("PDF has no readable text")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 🚨 Check chunks
    if not chunks:
        raise ValueError("No text chunks created from PDF")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    return db


# ---- Graph builder ----
def build_graph(db):

    def retrieve(state: State):
        retriever = db.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(state["query"])

        docs = [d for d in docs if len(d.page_content.strip()) > 30]

        context = "\n\n".join([doc.page_content for doc in docs])

        return {
            "context": context,
            "sources": docs
        }

    def generate(state: State):
        if not state["context"].strip():
            return {"answer": "Not found in document"}

        prompt = f"""
You are an intelligent assistant.

Use ONLY the context.
If not found → say "Not found in document".
Answer in 3-5 lines.

Context:
{state['context']}

Question:
{state['query']}

Answer:
"""

        response = ollama.chat(
            model="phi",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response["message"]["content"].strip()

        if len(answer) < 10:
            answer = "Not found in document"

        return {"answer": answer}

    builder = StateGraph(State)

    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()