import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

st.title("🧠 Smart RAG App (Specific Answers)")

uploaded_file = st.file_uploader("Upload a text file")

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

    db = FAISS.from_texts(chunks, embeddings)

    query = st.text_input("Ask a question")

    if query:
        docs = db.similarity_search(query, k=2)

        # Combine context
        context = " ".join([doc.page_content for doc in docs])

        # Local LLM
        llm = Ollama(model="mistral")

        # 🔥 PROMPT ENGINEERING (THIS IS THE KEY)
        prompt = f"""
        Answer the question based only on the context below.
        Give a short, clear, and specific answer (1-2 sentences).

        Context:
        {context}

        Question:
        {query}
        """

        answer = llm.invoke(prompt)

        st.subheader("📌 Answer")
        st.write(answer)