RAG-Based Document Q&A System

A fully offline Retrieval-Augmented Generation (RAG) application that enables semantic PDF search and context-aware question answering using local Large Language Models (LLMs).

Built using LangGraph, FAISS, HuggingFace Embeddings, Ollama, and Streamlit, the system processes uploaded PDFs, retrieves relevant document chunks through vector similarity search, and generates accurate answers without requiring internet connectivity.

🚀 Project Overview

Traditional document search systems rely heavily on keyword matching, which often fails to understand semantic meaning and context. This project solves that problem by combining:

Semantic vector search
Retrieval-Augmented Generation (RAG)
Local LLM inference
Workflow orchestration

The system allows users to:

Upload PDF documents
Automatically chunk and embed content
Perform semantic similarity search
Ask natural language questions
Receive context-aware answers generated completely offline

🔥 Key Features
Fully offline RAG pipeline
Semantic PDF search using vector embeddings
Efficient similarity retrieval with FAISS
Local LLM inference using Ollama
Workflow orchestration with LangGraph
Interactive Streamlit-based UI
Real-time question answering
Modular and scalable architecture


🛠️ Tech Stack
Python
LangGraph
FAISS
HuggingFace Sentence Transformers
Ollama
Streamlit
PyPDF
LangChain


🧠 System Architecture
1. PDF Upload & Parsing
Users upload PDF documents through the Streamlit interface
Text is extracted and cleaned for processing
2. Document Chunking
Large documents are split into smaller semantic chunks
Improves retrieval precision and context handling
3. Embedding Generation
Sentence-transformer embeddings generated using HuggingFace models
Converts text chunks into dense vector representations
4. Vector Storage with FAISS
Embeddings stored in FAISS vector database
Enables fast and scalable similarity search
5. Retrieval Pipeline
User query converted into embeddings
Most relevant document chunks retrieved using cosine similarity
6. Response Generation
Retrieved context passed to local LLM through Ollama
Generates context-aware answers without external APIs
7. Workflow Orchestration
LangGraph manages modular execution flow
Improves maintainability and scalability


📊 Key Highlights
Fully offline execution with no dependency on cloud APIs
Supports semantic understanding instead of keyword matching
Faster retrieval using optimized vector indexing
Modular workflow design for scalability
Real-time interaction through Streamlit interface
