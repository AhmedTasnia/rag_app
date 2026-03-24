import re

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="RAG App", layout="centered")

st.title("📄 RAG App")


def normalize_question(text: str) -> str:
    return " ".join(text.strip().lower().split())


# Add your fixed question-answer pairs here.
SPECIFIC_QA = {
    normalize_question("what is your name?"): "My name is RAG App Assistant.",
    normalize_question("who made you?"): "I was created inside this Streamlit RAG app.",
}


def get_specific_answer(query: str) -> str | None:
    normalized_query = normalize_question(query)

    if normalized_query in SPECIFIC_QA:
        return SPECIFIC_QA[normalized_query]

    for fixed_q, fixed_a in SPECIFIC_QA.items():
        if fixed_q in normalized_query or normalized_query in fixed_q:
            return fixed_a

    return None


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def best_sentence_answer(query: str, docs: list) -> str:
    query_tokens = tokenize(query)
    best_sentence = ""
    best_score = -1

    for doc in docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content)
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if not sentence_clean:
                continue

            sentence_tokens = tokenize(sentence_clean)
            overlap = len(query_tokens & sentence_tokens)

            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence_clean

    if best_sentence:
        return best_sentence

    return docs[0].page_content[:300].strip()


@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def build_vector_db(file_text: str) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    chunks = text_splitter.split_text(file_text)

    if not chunks:
        raise ValueError("The uploaded file is empty after text splitting.")

    return FAISS.from_texts(chunks, get_embeddings())

# Upload file
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

# User input is available even before upload so fixed Q/A can always work.
query = st.text_input("Ask a question")

if query:
    specific_answer = get_specific_answer(query)
    if specific_answer:
        st.subheader("📌 Answer")
        st.write(specific_answer)
        st.stop()

if uploaded_file is not None:
    try:
        text = uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        st.error("Could not decode the file. Please upload a UTF-8 .txt file.")
        st.stop()

    st.success("File uploaded successfully ✅")

    if query:
        try:
            db = build_vector_db(text)
        except Exception as exc:
            st.error(f"Failed to build search index: {exc}")
            st.stop()

        docs = db.similarity_search(query, k=2)

        if not docs:
            st.subheader("📌 Answer")
            st.write("No relevant context found in the uploaded file.")
            st.stop()

        relevant_text = " ".join(doc.page_content for doc in docs)

        sentences = relevant_text.split(".")
        short_answer = ". ".join(s.strip() for s in sentences if s.strip())
        short_answer = ". ".join(short_answer.split(". ")[:2])

        if not short_answer:
            short_answer = relevant_text[:200]

        st.subheader("📌 Answer")
        st.write(short_answer.strip() + ".")

        with st.expander("🔍 Source Context"):
            for i, doc in enumerate(docs):
                st.write(f"Chunk {i+1}:")
                st.write(doc.page_content)
                st.write("---")
elif query:
    st.info("Upload a text file for RAG search, or ask one of the fixed questions.")