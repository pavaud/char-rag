# RAG Chatbot with Streamlit

A conversational chatbot that lets users upload a PDF and ask questions about its content through a chat-like interface. The chatbot retrieves context-aware answers from the document.

## Features

- **PDF Upload**: Upload a PDF document through the sidebar.
- **Conversational Q&A**: Ask questions and receive enriched answers based on the document's content.
- **Efficient Retrieval**: Uses embeddings to find relevant text chunks for each question.

## Prerequisites

- `uv` package manager must be insalled

## Setup

1. **Install dependencies**

   ```bash
   uv sync
   ```

1. **Run Streamlit app**

    ```bash
    streamlit run ./src/chat.py
    ```

    Access the app here: `http://localhost:8501`.

## How It Works

- **PDF Processing**: Extracts and splits text using LangChain.
- **Embedding and Storage**: Embeds text chunks with Hugging Face and stores them in Chroma for similarity search.
- **Question-Answering**: Uses the top matching chunks to generate enriched answers.
