import streamlit as st
import os
from rag_graph import create_db, build_graph

st.set_page_config(page_title="Document Q&A", layout="wide")

st.title("📄 AI Document Q&A System (Upload PDF)")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully")

    # Create DB + Graph (cache)
    if "graph" not in st.session_state:
        db = create_db("temp.pdf")
        st.session_state.graph = build_graph(db)

    query = st.text_input("Ask a question:")

    if query:
        with st.spinner("Analyzing..."):
            result = st.session_state.graph.invoke({"query": query})

        answer = result["answer"]
        sources = result["sources"]

        if answer.lower().startswith("not found"):
            st.warning("Answer not found in document")
        else:
            st.success("Answer generated")

        st.subheader("📌 Answer")
        st.write(answer)

        st.subheader("📚 Sources")
        for i, doc in enumerate(sources, 1):
            st.markdown(f"**{i}.** {doc.page_content[:200]}...")


try:
    if "graph" not in st.session_state:
        db = create_db("temp.pdf")
        st.session_state.graph = build_graph(db)
except Exception as e:
    st.error(f"Error: {str(e)}")
