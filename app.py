import streamlit as st
from rag import build_rag

st.set_page_config(
    page_title="Mody AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Chat With PDF")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    # نحفظ الملف مؤقتاً
    pdf_path = f"books/{uploaded_file.name}"

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF Uploaded Successfully ✅")

    # بناء الـ RAG
    rag_chain = build_rag(pdf_path)

    question = st.text_input("Ask your question")

    if st.button("Ask"):

        response = rag_chain.invoke(
            {
                "input": question
            }
        )

        st.subheader("Answer")

        st.write(response["answer"])