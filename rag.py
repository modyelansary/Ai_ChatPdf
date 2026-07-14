from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from pathlib import Path
import os


def build_rag(pdf_path):

    # اسم الكتاب
    book_name = Path(pdf_path).stem

    # مكان قاعدة البيانات
    db_path = f"./database/{book_name}"

    # Embedding Model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # لو قاعدة البيانات موجودة افتحها
    if os.path.exists(db_path):

        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embedding,
        )

    else:

        # Load PDF
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        # Create Vector DB
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=db_path,
        )

    # Retriever
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 100}
    )

    # LLM
    llm = ChatOllama(
        model="llama3.2:3b"
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
You are an AI assistant.

Answer ONLY using the provided context.

If the answer is not found in the context, say:

"I don't have enough information in the provided document."

Context:
{context}

Question:
{input}

Answer:
""")

    # Chain
    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    rag_chain = create_retrieval_chain(
        retriever,
        document_chain
    )

    return rag_chain