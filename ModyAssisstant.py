from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)
from pathlib import Path





# Load PDF
loader = PyMuPDFLoader("book.pdf")
documents = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

# Embedding
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector Database

pdf_path = "book.pdf"
book_name = Path(pdf_path).stem

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory=f"./database/{book_name}"
)





# Retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k":3}
)


prompt = ChatPromptTemplate.from_template("""
Answer the question using ONLY the provided context.

If the answer is not in the context, say:
"I don't have enough information in the provided document."

Context:
{context}

Question:
{input}

Answer:
""")

# LLM
llm = ChatOllama(
    model="llama3.2:3b"
)

# Chain
document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    document_chain
)

# Ask
while True:

    question = input("\nAsk: ")

    if question.lower() == "exit":
        break

    response = rag_chain.invoke(
        {
            "input": question
        }
    )

    print("\nAnswer:\n")
    print(response["answer"])