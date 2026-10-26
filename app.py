
import os
import uuid

import gradio as gr
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

# Keep FAISS objects outside Gradio state.
# Gradio stores only the string ID.
vector_stores = {}

# Reuse the embedding model instead of loading it repeatedly.
embeddings_model = None


# --------------------------------------------------
# 2. Hugging Face embedding model
# --------------------------------------------------

def get_embeddings():
    global embeddings_model

    if embeddings_model is None:
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return embeddings_model


# --------------------------------------------------
# 3. Groq language model
# --------------------------------------------------

def get_llm():
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add your Groq API key to the .env file."
        )

    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0,
        max_retries=2,
    )


# --------------------------------------------------
# 4. Load PDF, split text, and create FAISS index
# --------------------------------------------------

def process_pdf(pdf_path):
    if not pdf_path:
        raise ValueError("Please upload a PDF first.")

    # Extract text and page metadata.
    documents = PyPDFLoader(pdf_path).load()

    if not documents:
        raise ValueError("No pages were found in this PDF.")

    # Divide document text into overlapping chunks.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError(
            "No readable text was found. "
            "Try a PDF containing selectable text."
        )

    # Convert chunks into vectors and store them in FAISS.
    vector_store = FAISS.from_documents(
        chunks,
        get_embeddings(),
    )

    return vector_store


# --------------------------------------------------
# 5. Process uploaded PDF
# --------------------------------------------------

def process_file(file_path, previous_store_id):
    if not file_path:
        return previous_store_id, "Please upload a PDF first."

    try:
        store = process_pdf(file_path)

        # Give this document index a unique identifier.
        store_id = uuid.uuid4().hex
        vector_stores[store_id] = store

        return (
            store_id,
            "PDF processed successfully! You can ask questions now.",
        )

    except Exception as exc:
        return None, f"PDF processing failed: {exc}"


# --------------------------------------------------
# 6. Retrieve relevant chunks and answer questions
# --------------------------------------------------

def answer_question(question, vector_store):
    if vector_store is None:
        return "Please upload and process a PDF first."

    if not question or not question.strip():
        return "Please enter a question."

    # Retrieve the three most relevant document chunks.
    results = vector_store.similarity_search(
        question,
        k=3,
    )

    if not results:
        return "I could not find relevant information in this document."

    # Include the PDF page number with each retrieved chunk.
    context = "\n\n".join(
        f"PDF page {doc.metadata.get('page', 0) + 1}:\n"
        f"{doc.page_content}"
        for doc in results
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful document question-answering assistant. "
                "Answer using only the provided document context. "
                "If the answer is not present in the context, say that "
                "you could not find it in the uploaded document. "
                "Do not invent facts.\n\n"
                "Document context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )

    # Generate the answer using Groq.
    response = (prompt | get_llm()).invoke(
        {
            "context": context,
            "question": question,
        }
    )

    answer = response.content

    # Handle either plain-text or list-based model responses.
    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", str(item))
            if isinstance(item, dict)
            else str(item)
            for item in answer
        )

    # Report the pages from which the retrieved context came.
    source_pages = sorted(
        {
            doc.metadata.get("page", 0) + 1
            for doc in results
        }
    )

    sources = ", ".join(str(page) for page in source_pages)

    return f"{answer}\n\nSources: PDF page(s) {sources}"


# --------------------------------------------------
# 7. Build Gradio interface
# --------------------------------------------------

def build_interface():

    with gr.Blocks(title="RAG DocuChat") as demo:

        gr.Markdown(
            """
            # RAG DocuChat

            Upload a PDF and ask questions about its content.

            This application uses retrieval-augmented generation (RAG)
            to find relevant passages before generating an answer.
            """
        )

        # Only a simple ID is stored in Gradio state.
        store_id_state = gr.State(value=None)

        pdf_input = gr.File(
            label="Upload your PDF",
            file_types=[".pdf"],
            type="filepath",
        )

        process_button = gr.Button(
            "Process PDF",
            variant="primary",
        )

        status = gr.Markdown()

        # Compatible with the installed Gradio version:
        # do not pass the unsupported type argument.
        chatbot = gr.Chatbot(
            label="Document Chat",
            height=400,
        )

        question = gr.Textbox(
            label="Ask a question",
            placeholder="Example: What is this document about?",
            lines=2,
        )

        with gr.Row():
            send_button = gr.Button(
                "Ask",
                variant="primary",
            )
            clear_button = gr.Button("Clear Chat")

        # Process PDF and preserve the existing ID if no new file is chosen.
        process_button.click(
            fn=process_file,
            inputs=[pdf_input, store_id_state],
            outputs=[store_id_state, status],
        )

        # Handle a user question and update message history.
        def chat(message, history, store_id):
            history = list(history or [])

            if not message or not message.strip():
                return history, ""

            store = (
                vector_stores.get(store_id)
                if store_id
                else None
            )

            try:
                answer = answer_question(message, store)

            except Exception as exc:
                answer = f"Sorry, an error occurred: {exc}"

            history.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

            history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            return history, ""

        send_button.click(
            fn=chat,
            inputs=[question, chatbot, store_id_state],
            outputs=[chatbot, question],
        )

        question.submit(
            fn=chat,
            inputs=[question, chatbot, store_id_state],
            outputs=[chatbot, question],
        )

        clear_button.click(
            fn=lambda: [],
            inputs=[],
            outputs=[chatbot],
        )

    return demo


# --------------------------------------------------
# 8. Start application
# --------------------------------------------------

if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
