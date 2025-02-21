# pip install -e .
# streamlit run Streamlit.py

import logging
import os
import dotenv
import pandas as pd
import asyncio
import hashlib
import streamlit as st
import json
from presidio_analyzer import RecognizerResult

from annotated_text import annotated_text
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
from Pinecone_LlamaIndex import loadDataPinecone, getResponse

try:
    st.set_page_config(
        page_title="GuardRAG",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception as e:
    pass 

st.title("GuardRAG")
st.subheader("Protecting private data in retrieval-augmented generation systems.")

st_logger = logging.getLogger('streamlit')
st_logger.setLevel(logging.INFO)

dotenv.load_dotenv()

database_file = None
index_name = "masters-thesis-index"

# SIDE BAR

pdf_records = list_pdf_records()
st_logger.info("PDF records loaded.")
pdf_options = [f"{record['id']} - {record['file_name']} - {record['file_hash']}" for record in pdf_records]
selected_file = st.sidebar.selectbox("Choose the data:", options=pdf_options)
st_logger.info(f"Selected file: {selected_file}")
if selected_file:
    selected_hash = selected_file.split(" - ")[2]  
    database_file = check_pdf_exists(selected_hash)  
    st_logger.info("Selected file from database")

uploaded_file = st.sidebar.file_uploader("Upload a PDF file:", type="pdf")

st_operator = st.sidebar.selectbox("Choose the privacy-preserving method:", options=["DELETE", "REPLACE with label", "REPLACE with synthetic data", "Differential privacy"])

st.sidebar.selectbox("Choose the RAG model:", options=["Simple RAG", "Chatbot RAG"])
st.sidebar.selectbox("Choose the LLM:", options=["gpt-4o-mini"])

if uploaded_file is not None:
    pdf_bytes = uploaded_file.read()
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    existing_file = check_pdf_exists(file_hash)
    st_logger.info(f"File hash: {file_hash}")

    if existing_file is not None:
        database_file = existing_file
        st_logger.info("Existing file found in the database.")

    if existing_file is None:
        st_logger.info("No existing file found, processing the uploaded PDF.")
        text_with_pii = asyncio.run(convert_pdf_to_text(pdf_bytes))
        st_logger.info("Converted PDF to text.")

        analyzer = analyzer_engine()
        st_analyze_results = analyze(
            text=text_with_pii,
            language="en",
            score_threshold=0.5,
            allow_list=[],
        )
        st_logger.info(f"Text analysis completed.{st_analyze_results}" )

        # Convert each RecognizerResult to a dictionary
        results_as_dicts = [result.to_dict() for result in st_analyze_results]

        # Serialize the list of dictionaries to a JSON string
        results_json = json.dumps(results_as_dicts, indent=2)

        # Log or display the JSON string
        st_logger.info(f"Text analysis results in JSON: {results_json}")
        st.write(results_json)
        
        text_pii_deleted = anonymize(
            text=text_with_pii,
            operator="redact", 
            analyze_results=st_analyze_results,
        )
        st_logger.info("Text with PII deleted.")
        st_logger.info(f"Labels: {text_pii_deleted} ")

        text_pii_labeled = anonymize(
            text=text_with_pii,
            operator="replace", 
            analyze_results=st_analyze_results,
        )
        st_logger.info("Text with PII labeled.")        
        st_logger.info(f"Labels: {text_pii_labeled} ")

        open_ai_params = OpenAIParams(
            openai_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-3.5-turbo-instruct",
            api_base=None,
            deployment_id="",
            api_version=None,
            api_type="openai",
        )
        text_pii_synthetic = create_fake_data(
            text_with_pii,
            st_analyze_results,
            open_ai_params,
        )   
        st_logger.info("Synthetic data created.")
        st_logger.info(f"Synthetic: {text_pii_synthetic} ")

        st_logger.info(f"loadDataPinecone: {uploaded_file.name} {file_hash} ")
        
        loadDataPinecone(
            index_name=index_name,
            text=text_with_pii,
            file_name=uploaded_file.name,
            file_hash=file_hash,
            text_type="text_with_pii"
        )
        loadDataPinecone(
            index_name=index_name,
            text=text_pii_deleted.text,
            file_name=uploaded_file.name,
            file_hash=file_hash,
            text_type="text_pii_deleted"
        )
        loadDataPinecone(
            index_name=index_name,
            text=text_pii_labeled.text,
            file_name=uploaded_file.name,
            file_hash=file_hash,
            text_type="text_pii_labeled"
        )
        loadDataPinecone(
            index_name=index_name,
            text=text_pii_synthetic,
            file_name=uploaded_file.name,
            file_hash=file_hash,
            text_type="text_pii_synthetic"
        )
        
        insert_pdf_record(uploaded_file.name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted.text, text_pii_labeled.text, text_pii_synthetic, results_json)
        st_logger.info("PDF record inserted into the database.")

# MAIN PANNEL

if database_file is not None:
    st_logger.info("Database file is available.")
    col1, col2 = st.columns(2)

    col1.text_area(
        label="Input: text containing private information", 
        value=database_file['text_with_pii'], 
        height=400, 
        key="text_input",
        label_visibility="visible" # visible, hidden, collapsed
    )
    st_logger.info("Input text area displayed.")

    if st_operator == "DELETE": 
        with col2:
            st.text_area(
                label="Output: private information deleted",
                value=database_file['text_pii_deleted'],
                height=400,
                label_visibility="visible"
            )
            st_logger.info("Output for DELETE method displayed.")
    elif st_operator == "REPLACE with label":
        with col2:
            st.text_area(
                label="Output: private information replaced by labels",
                value=database_file['text_pii_labeled'],
                height=400,
                label_visibility="visible"
            )
            st_logger.info("Output for REPLACE with label method displayed.")
    elif st_operator == "REPLACE with synthetic data":
        with col2:
            st.text_area(
                label="Output: private information replaced by synthetic data",
                value=database_file['text_pii_synthetic'],
                height=400,
                label_visibility="visible"
            )
            st_logger.info("Output for REPLACE with synthetic data method displayed.")

    # Assuming database_file['details'] is a JSON string
    st_analyze_results = [RecognizerResult(**item) for item in json.loads(database_file['details'])]

    # Put PII entities in a collapsed expander
    with st.expander("Detected PII Entities", expanded=False):
        annotated_tokens = annotate(text=database_file['text_with_pii'], analyze_results=st_analyze_results)
        annotated_text(*annotated_tokens)

    # Put findings table in a collapsed expander
    with st.expander("Detailed Findings", expanded=False):
        if database_file['details']:
            df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
            # Create the text slice once, then reuse it.
            df["Text"] = [database_file['text_with_pii'][res.start:res.end] for res in st_analyze_results]
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
else:
    st.warning("No data available to display.")

st_logger.info("Waiting for user input...")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    st_logger.info("Get Answer button clicked.")
    if question and database_file:

        if st_operator == "DELETE": 
            text_type_filter = "text_pii_deleted"
        elif st_operator == "REPLACE with label":
            text_type_filter = "text_pii_labeled"
        elif st_operator == "REPLACE with synthetic data":
            text_type_filter = "text_pii_synthetic"

        response_with_pii = getResponse(index_name, question, [database_file['file_hash'], "text_with_pii"])
        response_without_pii = getResponse(index_name, question, [database_file['file_hash'], text_type_filter])
        
        st_logger.info("Answers retrieved from query engine.")

        col1, col2 = st.columns(2)

        col1.text_area(
            label="Response based on the text containing private information", 
            value=response_with_pii, 
            height=200, 
            key="text_input_with_pii",
            label_visibility="visible" # visible, hidden, collapsed
        )

        col2.text_area(
            label="Response based on the text with applied privacy-preverving method", 
            value=response_without_pii, 
            height=200, 
            key="text_input_without_pii",
            label_visibility="visible" # visible, hidden, collapsed
        )
    else:
        st.warning("Please select at least one PDF or upload a new one and enter a question.")