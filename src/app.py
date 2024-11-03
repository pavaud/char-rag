import tempfile

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from streamlit import session_state as stss
from streamlit.runtime.uploaded_file_manager import UploadedFile
from transformers import pipeline


def init_session_state() -> None:
    stss.setdefault("already_uploaded", False)
    stss.setdefault("messages", [])
    stss.setdefault("uploaded_file", None)
    stss.setdefault("info_container", None)


def display_message(content: str, role: str) -> None:
    stss.messages.append({"role": role, "content": content})
    st.chat_message(role).write(content)


def compute_answer(question: str) -> str:
    # Encode question and retrieve top relevant chunks
    question_embedding = embedding_function.embed_query(question)
    relevant_docs = chroma_db.similarity_search_by_vector(question_embedding, k=10)
    combined_context = " ".join([doc.page_content for doc in relevant_docs])

    # Infer answer
    qa_pipeline = pipeline(
        task="question-answering",
        model="distilbert-base-uncased-distilled-squad",
    )
    answer = qa_pipeline(question=question, context=combined_context)

    return answer["answer"]


def display_chat_history() -> None:
    for message in stss.messages:
        st.chat_message(message["role"]).write(message["content"])


def display_prompt() -> None:
    if question := st.chat_input("Ask a question about the PDF:"):
        if stss.uploaded_file:
            display_message(question, "user")
            answer = compute_answer(question)
            display_message(answer, "assistant")
        else:
            stss.info_container.error("Upload a file before asking questions.")


def display_chat() -> None:
    display_chat_history()
    display_prompt()


def generate_embeddings(uploaded_file: UploadedFile) -> None:
    """Generates embeddings from pdf file and saves it in database."""

    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_file_path = tmp_file.name

    # Load and split PDF into text chunks
    loader = PyMuPDFLoader(temp_file_path)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(document)

    # Generate embeddings
    for chunk in chunks:
        chroma_db.add_texts([chunk.page_content])


st.title("RAG Chatbot")

init_session_state()

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_db = Chroma(
    collection_name="pdf_store",
    embedding_function=embedding_function,
    persist_directory="chromadb",
)

# Sidebar
with st.sidebar:
    stss.uploaded_file = st.file_uploader("**Upload a PDF file**", type="pdf")
    stss.info_container = st.container()

if stss.uploaded_file and not stss.already_uploaded:
    stss.already_uploaded = True
    generate_embeddings(stss.uploaded_file)

# Chat
display_chat()
