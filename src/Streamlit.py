# pip install -e .
# streamlit run Streamlit.py

import logging
import hashlib
import dotenv
import pandas as pd
import streamlit as st
import json
import asyncio
from presidio_analyzer import RecognizerResult
from PDF_reader import convert_pdf_to_text

from annotated_text import annotated_text
from Storage import list_records, retrieve_record_by_name, retrieve_record_by_hash
from Presidio_helpers import (
    analyze,
    annotate,
    analyzer_engine,
)    
from Presidio_OpenAI import OpenAIParams
from Pinecone_LlamaIndex import getResponse
from Data_loader import load_data, load_data_de
from question_generator import generate_questions_pii, evaluation

try:
    st.set_page_config(
        page_title="GuardRAG",
        layout="wide", #"centered" 
        initial_sidebar_state="collapsed" #"expanded",
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

# MAIN PANNEL

col1, col2, col3 = st.columns([1, 1, 2])

# INPUT - COLUMN 1
db_records = list_records()
st_logger.info("DB records loaded.")
db_options = [record['file_name'] for record in db_records]
selected_file = col1.selectbox("Choose a file:", options=db_options, index=None)
st_logger.info(f"Selected file: {selected_file}")
if selected_file is not None:
    database_file = retrieve_record_by_name(selected_file)
    st_logger.info("Record retrieved from database")

# INPUT - COLUMN 2
uploaded_file = col2.file_uploader("Upload a file:", type=["pdf", "txt"])
language = col2.selectbox("Select language:", options=["en", "de"], index=None, key="file language")  # Language selection box
if uploaded_file is not None and not language:
    st.error("Please enter the language of your file.")
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1]
    file_bytes = uploaded_file.read()
    if file_extension == "pdf":
        text_with_pii = asyncio.run(convert_pdf_to_text(file_bytes))  # Await the async function
    elif file_extension == "txt":
        text_with_pii = file_bytes.decode("utf-8")  # Decode bytes to string for TXT file
    file_hash = hashlib.sha256(text_with_pii.encode("utf-8")).hexdigest()  # Compute hash from string
    if retrieve_record_by_hash(file_hash) is not None:
        database_file = retrieve_record_by_hash(file_hash)
    elif language == "de":
        database_file = load_data_de(text_with_pii, uploaded_file.name, file_hash, uploaded_file.read(), index_name, st_logger)
    else:        
        database_file = load_data(text_with_pii, uploaded_file.name, file_hash, uploaded_file.read(), index_name, st_logger)

# INPUT - COLUMN 3
with col3:
    text_with_pii = st.text_area("Type your text here:", height=68)
    user_file_name = st.text_input("Enter a name so the text can be saved:")
    language = st.selectbox("Select language:", options=["en", "de"], index=None, key="typed text language")  # Language selection box

    if st.button("Send"):
        if not user_file_name:  # Check if user_file_name is empty
            st.error("Please enter a name under which the text can be saved.")  # Display warning
        elif not language:
            st.error("Please enter the language of your text.")  # Display warning
        else:
            st_logger.info(f"User input sent: {text_with_pii}")
            st_logger.info(f"User file name: {user_file_name}")
            st_logger.info(f"Selected language: {language}")  # Log the selected language
            file_bytes = text_with_pii.encode("utf-8")  # Convert the text to bytes
            file_hash = hashlib.sha256(file_bytes).hexdigest()  # Compute hash from bytes
            if retrieve_record_by_hash(file_hash) is not None:
                database_file = retrieve_record_by_hash(file_hash)
            elif language == "de":
                database_file = load_data_de(text_with_pii, user_file_name, file_hash, file_bytes, index_name, st_logger)
            else:
                database_file = load_data(text_with_pii, user_file_name, file_hash, file_bytes, index_name, st_logger)

# PII FINDINGS 

if database_file is not None:
    st_logger.info("Database file is available.")

    st_analyze_results = [RecognizerResult(**item) for item in json.loads(database_file['details'])]
    with st.expander("Text with detected Personally Identifiable Information (PII)", expanded=True):
        annotated_tokens = annotate(text=database_file['text_with_pii'], analyze_results=st_analyze_results)
        annotated_text(*annotated_tokens)

    with st.expander("Detailed Findings", expanded=False):
        if database_file['details']:
            df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
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
            st.text("No Personally identifiable information (PII) found")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    col1.text_area(
        label="Text with PII deleted",
        value=database_file['text_pii_deleted'],
        height=400,
        label_visibility="visible"
    )
    st_logger.info("Output for DELETE method displayed.")

    col2.text_area(
        label="Text with PII replaced by labels",
        value=database_file['text_pii_labeled'],
        height=400,
        label_visibility="visible"
    )
    st_logger.info("Output for REPLACE with label method displayed.")

    col3.text_area(
        label="Text with PII replaced by synthetic data",
        value=database_file['text_pii_synthetic'],
        height=400,
        label_visibility="visible"
    )
    st_logger.info("Output for REPLACE with synthetic data method displayed.")

    col4.text_area(
        label="Text transformed through differential privacy method",
        value=database_file['text_pii_dp_prompt'],
        height=400,
        label_visibility="visible"
    )
    st_logger.info("Output for differential privacy method displayed.")

else:
    st.warning("Please choose an existing file, upload a new file or create a new file")

# questions = generate_questions_pii(database_file['text_with_pii'], st_analyze_results)
# with st.expander("Example Questions", expanded=False):
#    for question in questions:
#       st.write(question)

# RAG 

st_logger.info("Waiting for user input...")
question = st.text_input("Enter your question:")

# ANSWERS 

if st.button("Get Answer"):
    st_logger.info("Get Answer button clicked.")

    if question and database_file:

        (response_with_pii, nodes_response_with_pii, evaluation_with_pii) = getResponse(index_name, question, [database_file['file_hash'], "text_with_pii"])
        (response_deleted, nodes_response_deleted, evaluation_deleted) = getResponse(index_name, question, [database_file['file_hash'], "text_pii_deleted"])
        (response_labeled, nodes_response_labeled, evaluation_labeled) = getResponse(index_name, question, [database_file['file_hash'], "text_pii_labeled"])
        (response_synthetic, nodes_response_synthetic, evaluation_synthetic) = getResponse(index_name, question, [database_file['file_hash'], "text_pii_synthetic"])
        (response_dp, nodes_response_dp, evaluation_dp) = getResponse(index_name, question, [database_file['file_hash'], "text_pii_dp_prompt"])
        st_logger.info("Answers and nodes succesfully retrieved from query engine.")

        st.text_area(
            label="Response based on the text containing PII", 
            value=response_with_pii, 
            height=200, 
            key="text_input_with_pii",
            label_visibility="visible" # visible, hidden, collapsed
        )

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        col1.text_area(
            label="Response based on the text with PII deleted", 
            value=response_deleted, 
            height=200, 
            key="text_pii_deleted",
            label_visibility="visible" # visible, hidden, collapsed
        )
        col2.text_area(
            label="Response based on the text with PII replaced by labels", 
            value=response_labeled, 
            height=200, 
            key="text_pii_labeled",
            label_visibility="visible" # visible, hidden, collapsed
        )
        col3.text_area(
            label="Response based on the text with PII replaced by synthetic data", 
            value=response_synthetic, 
            height=200, 
            key="text_pii_synthetic",
            label_visibility="visible" # visible, hidden, collapsed
        )
        col4.text_area(
            label="Response based on the text with PII transformed with differential private method", 
            value=response_dp, 
            height=200, 
            key="text_pii_dp_prompt",
            label_visibility="visible" # visible, hidden, collapsed
        )

        # NODES
        with st.expander("Nodes retrieved from the text containing PII", expanded=False):
            st.write(nodes_response_with_pii) 

        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            with st.expander("Nodes retrieved from the text with deleted PII", expanded=False):
                st.write(nodes_response_deleted) 
        with col2:
            with st.expander("Nodes retrieved from the text with PII replaced by labels", expanded=False):
                st.write(nodes_response_labeled) 
        with col3:
            with st.expander("Nodes retrieved from the text with PII replaced by syntehtic data", expanded=False):
                st.write(nodes_response_synthetic) 
        with col4:
            with st.expander("Nodes retrieved from the text with applied privacy-preserving method", expanded=False):
                st.write(nodes_response_dp) 

        # PII IN RESPONSE
        def create_pii_table(text):
            analyzer = analyzer_engine()
            st_analyze_results = analyze(
                text=text,
                language="en",
                score_threshold=0.5,
                allow_list=[],
            )
            if st_analyze_results:
                df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
                df["Text"] = [text[res.start:res.end] for res in st_analyze_results]
                df_subset = df[["entity_type", "Text", "start", "end", "score"]].rename(
                    {
                        "entity_type": "Entity type",
                        "start": "Start",
                        "end": "End",
                        "score": "Confidence",
                    },
                    axis=1,
                )
                return df_subset
            return pd.DataFrame(columns=["Entity type", "Text", "Start", "End", "Confidence"])  # Return an empty DataFrame
 
        with st.expander("PII in response based on raw text", expanded=False):
            st.dataframe(create_pii_table(str(response_with_pii)).reset_index(drop=True), use_container_width=True)
            
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            with st.expander("PII in response based on text with PII deleted", expanded=False):
                if response_deleted is not None:
                    st.dataframe(create_pii_table(str(response_deleted)).reset_index(drop=True), use_container_width=True)
        with col2:
            with st.expander("PII in response based on text with PII labeled", expanded=False):
                if response_labeled is not None:
                    st.dataframe(create_pii_table(str(response_labeled)).reset_index(drop=True), use_container_width=True)
        with col3:
            with st.expander("PII in response based on text with syntethic PII", expanded=False):
                if response_synthetic is not None:
                    st.dataframe(create_pii_table(str(response_synthetic)).reset_index(drop=True), use_container_width=True)
        with col4:
            with st.expander("PII in response based on differentially private text", expanded=False):
                if response_dp is not None:
                    st.dataframe(create_pii_table(str(response_dp)).reset_index(drop=True), use_container_width=True)

        # EVALUATION 
        evaluator_types = list(evaluation_with_pii.keys())
        eval_results_with_pii = [evaluation_with_pii[evaluator] for evaluator in evaluator_types]
        eval_results_deleted = [evaluation_deleted[evaluator] for evaluator in evaluator_types]
        eval_results_labeled = [evaluation_labeled[evaluator] for evaluator in evaluator_types]
        eval_results_synthetic = [evaluation_synthetic[evaluator] for evaluator in evaluator_types]
        eval_results_dp = [evaluation_dp[evaluator] for evaluator in evaluator_types]

        evaluation_df = pd.DataFrame({
            'Evaluator': evaluator_types,
            'Response with PII': eval_results_with_pii,
            'Response PII deleted': eval_results_deleted,
            'Response PII labeled': eval_results_labeled,
            'Response PII synthetic': eval_results_synthetic,
            'Response PII DP': eval_results_dp        
        })

        # evaluation_df = evaluation(questions, database_file['file_hash'])
        # Display the evaluation results as a table
        st.subheader("Evaluation Results")
        st.dataframe(evaluation_df)