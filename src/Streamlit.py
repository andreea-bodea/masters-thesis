
# pip install -e .
# streamlit run Streamlit.py

import os
import dotenv
import pandas as pd
import asyncio
import hashlib
import streamlit as st

from annotated_text import annotated_text
from Cache import Cache
from Storage import check_pdf_exists, insert_pdf_record, list_pdf_records
from PDF_reader import convert_pdf_to_text
from Presidio_helpers import (
    analyze,
    anonymize,
    annotate,
    create_fake_data,
    analyzer_engine,
)
from Presidio_OpenAI import OpenAIParams
from Presidio_anoymization import anonymize_text
from Pinecone_RAG import query_engine 

st.set_page_config(
    page_title="GuardRAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("GuardRAG")
st.subheader("Protecting private data in retrieval-augmented generation systems.")

dotenv.load_dotenv()

# List and select PDFs
pdf_records = list_pdf_records()
pdf_options = [f"{record['id']} - {record['file_name']}" for record in pdf_records]
selected_pdfs = st.sidebar.multiselect("Choose the data:", options=pdf_options)

uploaded_file = st.sidebar.file_uploader("Upload a PDF file:", type="pdf")

st_operator = st.sidebar.selectbox("Choose the privacy-preserving method:", options=["DELETE", "REPLACE with label", "REPLACE with synthetic data", "Differential privacy"])
if st_operator == "DELETE":
    st_operator = "redact"
elif st_operator == "REPLACE with label":
    st_operator = "replace"
elif st_operator == "REPLACE with synthetic data":
    open_ai_params = OpenAIParams(
        openai_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo-instruct",
        api_base=None,
        deployment_id="",
        api_version=None,
        api_type="openai",
    )

st.sidebar.selectbox("Choose the RAG model:", options=["Simple RAG", "Chatbot RAG"])
st.sidebar.selectbox("Choose the LLM:", options=["gpt-4o-mini"])

if uploaded_file is not None:
    # Read the PDF file (as bytes)
    pdf_bytes = uploaded_file.read()

    # Compute a unique hash for the file (using SHA-256)
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Check if this PDF already exists in the database
    existing_record = check_pdf_exists(file_hash)
    if existing_record:
        st.info("This PDF already exists in the database.")
        st.text_area("Original Text", existing_record["extracted_text"], height=400)
        st.text_area("Anonymized Text", existing_record["anonymized_text"], height=400)
    if existing_record == False:
        # Convert PDF to text using asyncio to run the coroutine
        original_text = asyncio.run(convert_pdf_to_text(pdf_bytes))
        st.write("Original Text:")
        st.text_area("Original Text", original_text, height=400)

        # Anonymize the extracted text (anonymize_text returns a tuple: (original, anonymized))
        orig_text, anonymized_text = anonymize_text(original_text)
        st.write("### Anonymized Text")
        st.text_area("Anonymized Text", anonymized_text, height=400)

        # Insert the new PDF record into the database
        insert_pdf_record(uploaded_file.name, file_hash, pdf_bytes, original_text, anonymized_text)
        st.success("PDF inserted into the database.")

# MAIN PANNEL

analyzer = analyzer_engine()

# Create two columns for input and output.
col1, col2 = st.columns(2)

# Input Text Area with an accessible (but visually hidden) label.
col1.subheader("Input: text containing private information")

st_text = col1.text_area(
    label="Input Text", 
    value="", 
    height=400, 
    key="text_input",
    label_visibility="collapsed"
)

try:
    st_analyze_results = analyze(
        text=st_text,
        language="en",
        score_threshold=0.5,
        allow_list=[],
    )

    # Display the processed output
    if st_operator == "redact": 
        with col2:
            st.subheader("Output: private information deleted")
            st_anonymize_results = anonymize(
                text=st_text,
                operator=st_operator,
                analyze_results=st_analyze_results,
            )
            st.text_area(
                label="Output",
                value=st_anonymize_results.text,
                height=400,
                label_visibility="collapsed"
            )
    elif st_operator == "replace":
        with col2:
            st.subheader("Output: private information replaced by labels")
            st_anonymize_results = anonymize(
                text=st_text,
                operator=st_operator,
                analyze_results=st_analyze_results,
            )
            st.text_area(
                label="Output",
                value=st_anonymize_results.text,
                height=400,
                label_visibility="collapsed"
            )
    elif st_operator == "REPLACE with synthetic data":
        with col2:
            st.subheader("Output: private information replaced by synthetic data")
            fake_data = create_fake_data(
                st_text,
                st_analyze_results,
                open_ai_params,
            )
            st.text_area(
                label="Output",
                value=fake_data,
                height=400,
                label_visibility="collapsed"
            )

    # Detailed findings
    with st.expander("Detailed findings", expanded=False):
        annotated_tokens = annotate(text=st_text, analyze_results=st_analyze_results)
        annotated_text(*annotated_tokens)

        if st_analyze_results:
            df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
            # Create the text slice once, then reuse it.
            df["Text"] = [st_text[res.start:res.end] for res in st_analyze_results]
            df_subset = df[["entity_type", "Text", "start", "end", "score"]].rename(
                {
                    "entity_type": "Entity type",
                    "start": "Start",
                    "end": "End",
                    "score": "Confidence",
                },
                axis=1,
            )
            st.dataframe(df_subset.reset_index(drop=True), use_container_width=True)
        else:
            st.text("No findings")

except Exception as e:
    st.error(f"An error occurred: {e}")


# Add a text input for the question
question = st.text_input("Enter your question:")

# Add a button to submit the question
if st.button("Get Answer"):
    if question and selected_pdfs:
        # Query the RAG system
        response = query_engine.query(question)
        st.write("Answer:")
        st.text_area("Answer", response, height=200)
    else:
        st.warning("Please select at least one PDF and enter a question.")
