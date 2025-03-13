import json
import os
import dotenv
from openai import OpenAI
import logging
from presidio_analyzer import RecognizerResult
from LLMDP import DPPrompt
from Storage import insert_record, retrieve_record_by_hash
from Presidio_helpers import analyze, anonymize, create_fake_data, analyzer_engine
from Presidio_OpenAI import OpenAIParams
from Pinecone_LlamaIndex import loadDataPinecone

dotenv.load_dotenv()

def load_data_de(text_with_pii, file_name, file_hash, file_bytes, index_name, st_logger):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18", 
        messages=[
        {"role": "system", "content": "You are a PII detection assistant. Identify all personally identifiable information (PII) in the provided German text and return it as a JSON object."},
        {"role": "user", "content": f"Finde PII in folgendem Text und gib eine JSON-Antwort zurück: '{text_with_pii}'"}
    ],
    temperature=0,
    response_format={"type": "json_object"}    
    )
    print(response.choices[0].message.content)
    
    """
    loadDataPinecone(
        index_name=index_name,
        text=text_with_pii,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_with_pii"
    )
    loadDataPinecone(
        index_name=index_name,
        text=text_pii_deleted.text,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_deleted"
    )
    loadDataPinecone(
        index_name=index_name,
        text=text_pii_labeled.text,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_labeled"
    )
    loadDataPinecone(
        index_name=index_name,
        text=text_pii_synthetic,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_synthetic"
    )

    loadDataPinecone(
        index_name=index_name,
        text=text_pii_dp_prompt,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_dp_prompt"
    )
    insert_record(file_name, file_hash, file_bytes, text_with_pii, text_pii_deleted.text, text_pii_labeled.text, text_pii_synthetic, text_pii_dp_prompt, results_json)
    st_logger.info("Document inserted into the database.")
    database_file = retrieve_record_by_hash(file_hash)

    return database_file
"""
    return None

def load_data(text_with_pii, file_name, file_hash, file_bytes, index_name, st_logger):
    analyzer = analyzer_engine()
    st_analyze_results = analyze(
        text=text_with_pii,
        language="en",
        score_threshold=0.5,
        allow_list=[],
    )
    st_logger.info(f"Text analysis completed.{st_analyze_results}")

    # Convert each RecognizerResult to a dictionary
    results_as_dicts = [result.to_dict() for result in st_analyze_results]

    # Serialize the list of dictionaries to a JSON string
    results_json = json.dumps(results_as_dicts, indent=2)
    st_logger.info(f"Text analysis results in JSON: {results_json}")

    text_pii_deleted = anonymize(
        text=text_with_pii,
        operator="redact",
        analyze_results=st_analyze_results,
    )
    st_logger.info("Text with PII deleted.")
    st_logger.info(f"Labels: {text_pii_deleted}")

    text_pii_labeled = anonymize(
        text=text_with_pii,
        operator="replace",
        analyze_results=st_analyze_results,
    )
    st_logger.info("Text with PII labeled.")
    st_logger.info(f"Labels: {text_pii_labeled}")

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

    # Initialize as an empty list to hold synthetic data results
    text_pii_synthetic_list = []

    for chunk in text_chunks:
        st_analyze_chunk_results = analyze(
            text=chunk,
            language="en",
            score_threshold=0.5,
            allow_list=[],
        )
        text_chunk_pii_synthetic_chunck = create_fake_data(
            chunk,
            st_analyze_chunk_results,
            open_ai_params,
        )
        text_pii_synthetic_list.append(text_chunk_pii_synthetic_chunck)

    text_pii_synthetic = ' '.join(text_pii_synthetic_list)

    st_logger.info("Synthetic data created.")
    st_logger.info(f"Synthetic: {text_pii_synthetic}")

    # def split_text_into_chunks(text, max_tokens=512):
        # Tokenize the text to count tokens
        # tokens = dpprompt.tokenizer.tokenize(text)
        # for i in range(0, len(tokens), max_tokens):
        #    yield dpprompt.tokenizer.convert_tokens_to_string(tokens[i:i + max_tokens])

    # Split the long text into manageable chunks
    # text_chunks = list(split_text_into_chunks(text_with_pii, max_tokens=512))

    dpprompt = DPPrompt(model_checkpoint="google/flan-t5-large")
    text_pii_dp_prompt = dpprompt.privatize(text_with_pii, epsilon=200)
    st_logger.info("Text with PII DP.")
    st_logger.info(f"Labels: {text_pii_dp_prompt}")

    st_logger.info(f"loadDataPinecone: {file_name} {file_hash}")

    loadDataPinecone(
        index_name=index_name,
        text=text_with_pii,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_with_pii"
    )
    loadDataPinecone(
        index_name=index_name,
        text=text_pii_deleted.text,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_deleted"
    )
    loadDataPinecone(
        index_name=index_name,
        text=text_pii_labeled.text,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_labeled"
    )
    loadDataPinecone(
        index_name=index_name,
        text=text_pii_synthetic,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_synthetic"
    )

    loadDataPinecone(
        index_name=index_name,
        text=text_pii_dp_prompt,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_dp_prompt"
    )
    insert_record(file_name, file_hash, file_bytes, text_with_pii, text_pii_deleted.text, text_pii_labeled.text, text_pii_synthetic, text_pii_dp_prompt, results_json)
    st_logger.info("Document inserted into the database.")
    database_file = retrieve_record_by_hash(file_hash)

    return database_file