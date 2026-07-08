import streamlit as st
import io
from contextlib import redirect_stdout

from level_23_f import run_rag_pipeline_level2

st.set_page_config(page_title="Scikit-Learn RAG")

st.title("Scikit-Learn RAG")

question = st.text_input("Ask a question")

if st.button("Ask"):

    if question.strip():

        output = io.StringIO()

        with st.spinner("Searching..."):
            with redirect_stdout(output):
                run_rag_pipeline_level2(question)

        st.text(output.getvalue())

    else:
        st.warning("Please enter a question.")