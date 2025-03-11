# pip install -e .
# streamlit run Streamlit.py

import logging
import dotenv
import pandas as pd
import streamlit as st
import json
from presidio_analyzer import RecognizerResult

from annotated_text import annotated_text
from Storage import check_document_exists, list_records
from Presidio_helpers import (
    analyze,
    annotate,
    analyzer_engine,
)    
from Presidio_OpenAI import OpenAIParams
from Pinecone_LlamaIndex import getResponse
from File_management import process_uploaded_file
from question_generator import generate_questions_pii, evaluation

try:
    st.set_page_config(
        page_title="GuardRAG",
        layout="wide",
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

# SIDE BAR


db_records = list_records()
st_logger.info("DB records loaded.")
db_options = [f"{record['id']} - {record['file_name']} - {record['file_hash']}" for record in db_records]
selected_file = st.sidebar.selectbox("Choose the data:", options=db_options)
st_logger.info(f"Selected file: {selected_file}")
if selected_file:
    selected_hash = selected_file.split(" - ")[2]  
    database_file = check_document_exists(selected_hash)  
    st_logger.info("Selected file from database")

uploaded_file = st.sidebar.file_uploader("Upload a file:", type=["pdf", "txt"])

st_operator = st.sidebar.selectbox("Choose the privacy-preserving method:", options=["DELETE", "REPLACE with label", "REPLACE with synthetic data", "Differential privacy"])

st.sidebar.selectbox("Choose the RAG model:", options=["Simple RAG", "Chatbot RAG"])
st.sidebar.selectbox("Choose the LLM:", options=["gpt-4o-mini"])

if uploaded_file is not None:
    if uploaded_file is not None:
        database_file = process_uploaded_file(uploaded_file, index_name, st_logger)

# MAIN PANNEL

if database_file is not None:
    st_logger.info("Database file is available.")

    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])     # col1, col2 = st.columns(2)

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
    elif st_operator == "Differential privacy":
        with col2:
            st.text_area(
                label="Output: private information - Differential privacy",
                value=database_file['text_pii_dp_prompt'],
                height=400,
                label_visibility="visible"
            )
            st_logger.info("Output for Differential privacy method displayed.")

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

# Generate questions based on detected PII entities
questions = generate_questions_pii(database_file['text_with_pii'], st_analyze_results)

# Display generated questions
with st.expander("Example Questions", expanded=False):
    for question in questions:
        st.write(question)

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
        elif st_operator == "Differential privacy":
            text_type_filter = "text_pii_dp_prompt"

        (response_with_pii, nodes_response_with_pii, evaluation_with_pii) = getResponse(index_name, question, [database_file['file_hash'], "text_with_pii"])
        (response_without_pii, nodes_response_without_pii, evaluation_without_pii) = getResponse(index_name, question, [database_file['file_hash'], text_type_filter])
        print(evaluation_with_pii)
        print(evaluation_without_pii)

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

        st_logger.info("Nodes retrieved from query engine.")

        col1, col2 = st.columns(2)

        # Display using st.expander for collapsible sections
        with col1:
            with st.expander("Nodes retrieved from the text containing private information", expanded=False):
                st.write(nodes_response_with_pii)  # Automatically formats the list

        with col2:
            with st.expander("Nodes retrieved from the text with applied privacy-preserving method", expanded=False):
                st.write(nodes_response_without_pii)  # Automatically formats the list

        analyzer = analyzer_engine()
        st_analyze_results = analyze(
            text=str(response_with_pii),
            language="en",
            score_threshold=0.5,
            allow_list=[],
        )
        with st.expander("Private Information in Response 1", expanded=False):
            if st_analyze_results:
                df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
                # Create the text slice once, then reuse it.
                df["Text"] = [str(response_with_pii)[res.start:res.end] for res in st_analyze_results]
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

        st_analyze_results_2 = analyze(
            text=str(response_without_pii),
            language="en",
            score_threshold=0.5,
            allow_list=[],
        )
        with st.expander("Private Information in Response 1", expanded=False):
            if st_analyze_results_2:
                df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results_2])
                # Create the text slice once, then reuse it.
                df["Text"] = [str(response_without_pii)[res.start:res.end] for res in st_analyze_results_2]
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

        # Create a list of evaluator types
        evaluator_types = list(evaluation_with_pii.keys())

        # Create a list of results for each evaluation
        results_with_pii = [evaluation_with_pii[evaluator] for evaluator in evaluator_types]
        results_without_pii = [evaluation_without_pii[evaluator] for evaluator in evaluator_types]

        # Create a single DataFrame
        evaluation_df = pd.DataFrame({
            'Evaluator': evaluator_types,
            'Result with PII': results_with_pii,
            'Result without PII': results_without_pii
        })

        # evaluation_df = evaluation(questions, database_file['file_hash'])
        # Display the evaluation results as a table
        st.subheader("Evaluation Results")
        st.dataframe(evaluation_df)

    else:
        st.warning("Please select at least one file or upload a new one and enter a question.")

