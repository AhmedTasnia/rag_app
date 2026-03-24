import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="RAG App", layout="centered")

st.title("📄 RAG App (Deployed - No AI)")

# Upload file
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")

    st.success("File uploaded successfully ✅")

    # Split text
    text_splitter = CharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB
    db = FAISS.from_texts(chunks, embeddings)

    # User input
    query = st.text_input("Ask a question")

    if query:
        docs = db.similarity_search(query, k=2)

        # 🔥 Extract only relevant sentences (IMPORTANT)
        relevant_text = " ".join([doc.page_content for doc in docs])

        # Try to make answer shorter
        sentences = relevant_text.split(".")
        short_answer = ". ".join(sentences[:2])  # first 2 sentences only

        st.subheader("📌 Answer")
        st.write(short_answer.strip() + ".")

        # Optional (for debugging / explanation)
        with st.expander("🔍 Source Context"):
            for i, doc in enumerate(docs):
                st.write(f"Chunk {i+1}:")
                st.write(doc.page_content)
                st.write("---")