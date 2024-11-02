import tempfile

import streamlit as st
from streamlit import session_state as stss

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline


if "already_uploaded" not in stss:
    stss.already_uploaded = False

if "messages" not in stss:
    stss.messages = []

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_db = Chroma("pdf_store", embedding_function=embedding_function)
qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert-base-uncased-distilled-squad",
)

st.title("RAG Chatbot")

# Sidebar

# Upload PDF file
with st.sidebar:
    uploaded_file = st.file_uploader("**Upload a PDF file**", type="pdf")
    info_container = st.container()

if uploaded_file and not stss.already_uploaded:
    stss.aleady_uploaded = True

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

# Chat

# Display message history
for message in stss.messages:
    if message["role"] == "user":
        st.chat_message("user").write(message["content"])
    else:
        st.chat_message("assistant").write(message["content"])

# User chat input
if question := st.chat_input("Ask a question about the PDF:"):
    if uploaded_file:

        stss.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        # Encode question and retrieve top relevant chunks
        question_embedding = embedding_function.embed_query(question)
        relevant_docs = chroma_db.similarity_search_by_vector(question_embedding, k=10)

        # Combine the content of top relevant chunks for enriched context
        combined_context = " ".join([doc.page_content for doc in relevant_docs])

        # Infer answer
        answer = qa_pipeline(question=question, context=combined_context)

        stss.messages.append({"role": "assistant", "content": answer["answer"]})
        st.chat_message("assistant").write(answer["answer"])
    else:
        info_container.error("Upload a file before asking questions.")
