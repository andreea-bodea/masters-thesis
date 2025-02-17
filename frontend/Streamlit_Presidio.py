"""Streamlit app for Presidio."""
import logging
import os

import dotenv
import pandas as pd
import streamlit as st
from annotated_text import annotated_text

from Presidio_OpenAI import OpenAIParams
from Presidio_helpers import (
    analyze,
    anonymize,
    annotate,
    create_fake_data,
    analyzer_engine,
)

st.set_page_config(
    page_title="Presidio demo",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "https://microsoft.github.io/presidio/",
    },
)

dotenv.load_dotenv()
logger = logging.getLogger("presidio-streamlit")

st_operator = st.sidebar.selectbox(
    "De-identification approach",
    ["redact", "replace", "synthesize"],
    index=1,
)

open_ai_params = None

logger.debug(f"Selected de-identification approach: {st_operator}")

def set_up_openai_synthesis():
    """Set up the OpenAI parameters for text synthesis using the API key from .env."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    # Hardcode other OpenAI settings as needed.
    return (
        "openai",         # api_type
        None,             # api_base
        "",               # deployment_id
        None,             # api_version
        openai_key,
        "gpt-3.5-turbo-instruct"  # model
    )

# Automatically set up synthesize parameters from .env.
if st_operator == "synthesize":
    (
        openai_api_type,
        st_openai_api_base,
        st_deployment_id,
        st_openai_version,
        st_openai_key,
        st_openai_model,
    ) = set_up_openai_synthesis()

    open_ai_params = OpenAIParams(
        openai_key=st_openai_key,
        model=st_openai_model,
        api_base=st_openai_api_base,
        deployment_id=st_deployment_id,
        api_version=st_openai_version,
        api_type=openai_api_type,
    )

# Main panel

# Use a spinner for loading the analyzer engine.
with st.spinner("Starting Presidio analyzer..."):
    analyzer = analyzer_engine()

demo_text = """Here are a few example sentences we currently support:

Hi, my name is David Johnson and I'm originally from Liverpool.
My credit card number is 4095-2609-9393-4932 and my crypto wallet id is 16Yeky6GMjeNkAiNcBY7ZhrLoMSgg1BoyZ.

On 11/10/2024 I visited www.microsoft.com and sent an email to test@presidio.site, from IP 192.168.0.1.

My passport: 191280342 and my phone number: (212) 555-1234.

This is a valid International Bank Account Number: IL150120690000003111111. Can you please check the status on bank account 954567876544? 

Kate's social security number is 078-05-1126. Her driver license? it is 1234567A."""

# Create two columns for input and output.
col1, col2 = st.columns(2)

# Input Text Area with an accessible (but visually hidden) label.
col1.subheader("Input")
st_text = col1.text_area(
    label="Input Text", 
    value=demo_text, 
    height=400, 
    key="text_input",
    label_visibility="collapsed"
)

try:
    with st.spinner("Analyzing text..."):
        st_analyze_results = analyze(
            text=st_text,
            language="en",
            score_threshold=0.5,
            allow_list=[],
        )

    # Display the processed output
    if st_operator == "synthesize":
        with col2:
            st.subheader("Output: text with private information replaced by synthetic data")
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
    elif st_operator == "replace":
        with col2:
            st.subheader("Output: text with private information replaced by labels")
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
    else:
        with col2: # delete
            st.subheader("Output: text with private information deleted")
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

    # Put PII entities in a collapsed expander
    with st.expander("Detected PII Entities", expanded=False):
        annotated_tokens = annotate(text=st_text, analyze_results=st_analyze_results)
        annotated_text(*annotated_tokens)

    # Put findings table in a collapsed expander
    with st.expander("Detailed Findings", expanded=False):
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
    logger.error("Error occurred:", exc_info=True)
    st.error(f"An error occurred: {e}")

