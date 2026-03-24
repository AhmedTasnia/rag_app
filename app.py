import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("RAG App (Deployed Version)")

uploaded_file = st.file_uploader("Upload a text file")

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

    st.write("File uploaded successfully ✅")

    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(chunks, embeddings)

    query = st.text_input("Ask a question")

    if query:
        docs = db.similarity_search(query)

        # No Ollama (Cloud safe)
        answer = docs[0].page_content

        st.write("Answer:")
        st.write(answer)