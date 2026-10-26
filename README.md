
# RAG DocuChat

A Retrieval-Augmented Generation (RAG) application that lets users upload PDF documents and ask questions about their content.

## Features
- PDF text extraction using PyPDFLoader
- Text chunking using RecursiveCharacterTextSplitter
- Semantic embeddings using Hugging Face
- Vector similarity search using FAISS
- AI-generated answers using Groq
- Interactive interface built with Gradio

## Technologies
Python, Gradio, LangChain, FAISS, Hugging Face, Groq API

## How it works
1. Upload a PDF document.
2. Extract and split the text into chunks.
3. Convert chunks into embeddings.
4. Retrieve relevant chunks using FAISS.
5. Generate an answer using the retrieved context.

## Running locally
Install the dependencies with:

pip install -r requirements.txt

Then run:

python app.py
