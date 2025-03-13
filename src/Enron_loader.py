import asyncio
import json
import os
import logging
import hashlib
import csv
import re
from presidio_analyzer import RecognizerResult
from LLMDP import DPPrompt
from Storage import check_document_exists, insert_record
from PDF_reader import convert_pdf_to_text
from Presidio_helpers import analyze, anonymize, create_fake_data, analyzer_engine
from Presidio_OpenAI import OpenAIParams
from Pinecone_LlamaIndex import loadDataPinecone

from Data_loader import process_uploaded_file

st_logger = logging.getLogger('enron')
st_logger.setLevel(logging.INFO)
        
def load_enron(uploaded_file_text, uploaded_file_name, index_name):
        file_bytes = None
        file_hash = uploaded_file_name
        text_with_pii = uploaded_file_text
        st_logger.info(f"File hash: {file_hash}")

        existing_file = check_document_exists(file_hash)
   
        if existing_file is not None:
            database_file = existing_file
            st_logger.info("Existing file found in the database.")
        else:
            st_logger.info("No existing file found, processing the uploaded file.")

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

            def split_text_into_chunks(text, max_words):
                """Split text into chunks of a specified maximum word count."""
                words = text.split()
                for i in range(0, len(words), max_words):
                    yield ' '.join(words[i:i + max_words])

            text_chunks = list(split_text_into_chunks(text_with_pii, max_words=2500))

            # List to hold synthetic data results
            text_pii_synthetic = ''

            for chunk in text_chunks:
                st_analyze_chunk_results = analyze(
                    text=chunk,
                    language="en",
                    score_threshold=0.5,
                    allow_list=[],
                )
                text_chunk_pii_synthetic = create_fake_data(
                    chunk,
                    st_analyze_chunk_results,
                    open_ai_params,
                )
                text_pii_synthetic.join(text_chunk_pii_synthetic)
            st_logger.info("Synthetic data created.")
            st_logger.info(f"Synthetic: {text_pii_synthetic} ")
            
            dpprompt = DPPrompt(model_checkpoint="google/flan-t5-large")
            text_pii_dp_prompt = dpprompt.privatize(text_with_pii, epsilon=200) 
            st_logger.info("Text with PII DP.")        
            st_logger.info(f"Labels: {text_pii_dp_prompt} ")


            def split_text_into_chunks(text, max_tokens=512):
                # Tokenize the text to count tokens
                tokens = dpprompt.tokenizer.tokenize(text)
                for i in range(0, len(tokens), max_tokens):
                    yield dpprompt.tokenizer.convert_tokens_to_string(tokens[i:i + max_tokens])

            # Split the long text into manageable chunks
            text_chunks = list(split_text_into_chunks(text_with_pii, max_tokens=512))

            # Use privatize_dp to process the chunks
            epsilon_value = 200  # Set your desired epsilon value

            st_logger.info(f"loadDataPinecone: {uploaded_file_name} ")
            
            loadDataPinecone(
                index_name=index_name,
                text=text_with_pii,
                file_name=uploaded_file_name,
                file_hash=file_hash,
                text_type="text_with_pii"
            )
            loadDataPinecone(
                index_name=index_name,
                text=text_pii_deleted.text,
                file_name=uploaded_file_name,
                file_hash=file_hash,
                text_type="text_pii_deleted"
            )
            loadDataPinecone(
                index_name=index_name,
                text=text_pii_labeled.text,
                file_name=uploaded_file_name,
                file_hash=file_hash,
                text_type="text_pii_labeled"
            )
            loadDataPinecone(
                index_name=index_name,
                text=text_pii_synthetic,
                file_name=uploaded_file_name,
                file_hash=file_hash,
                text_type="text_pii_synthetic"
            )

            loadDataPinecone(
                index_name=index_name,
                text=text_pii_dp_prompt,
                file_name=uploaded_file_name,
                file_hash=file_hash,
                text_type="text_pii_dp_prompt"
            )
            insert_record(uploaded_file_name, file_hash, file_bytes, text_with_pii, text_pii_deleted.text, text_pii_labeled.text, text_pii_synthetic, text_pii_dp_prompt, results_json)
            st_logger.info("Document inserted into the database.")
            database_file = check_document_exists(file_hash)

        return database_file

if __name__ == "__main__":
    file_path = "/Users/andreeabodea/ANDREEA/MT/emails_first_10.csv"
    index_name = "your_index_name"  # Replace with your actual index name

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the first row
        for row in csv_reader:
            # Assuming each row contains the text and file name
            uploaded_file_text = row[0]  # Adjust index based on your CSV structure
            message_id_match = re.search(r'Message-ID: <(.*?)>', uploaded_file_text)
            if message_id_match:
                message_id = message_id_match.group(1)
                uploaded_file_name = "Enron_" + message_id  # Adjust index based on your CSV structure
            else:
                uploaded_file_name = "Enron_Unknown"  # Fallback if Message-ID is not found
            # load_enron(uploaded_file_text, uploaded_file_name, index_name = "masters-thesis-index")
