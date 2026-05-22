import streamlit as st
from rag import initialize_rag, ask_rag

st.title("Document Assistant using RAG")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file and "chunks" not in st.session_state:

    chunks, chunks_embeddings = initialize_rag(uploaded_file)

    st.session_state["chunks"] = chunks
    st.session_state["embeddings"] = chunks_embeddings

if uploaded_file:

    chunks = st.session_state["chunks"]
    chunks_embeddings = st.session_state["embeddings"]

    st.success("Agent is ready!")

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("Submit") and question:

        result = ask_rag(
            question,
            chunks,
            chunks_embeddings
        )

        st.subheader("Answer")
        st.write(result["answer"])
