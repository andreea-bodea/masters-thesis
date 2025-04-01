import json
import os
import dotenv
from openai import OpenAI
import logging
from presidio_analyzer import RecognizerResult
from Database_management import insert_record, insert_record_complex, retrieve_record_by_hash
from Presidio_helpers import analyze, anonymize, create_fake_data, analyzer_engine
from Presidio_OpenAI import OpenAIParams
from Pinecone_LlamaIndex import loadDataPinecone
from Differential_Privacy import diff_privacy_dp_prompt, diff_privacy_diffractor

dotenv.load_dotenv()

def split_text_into_chunks(text, max_words):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i + max_words])
        chunks.append(chunk) 
    return chunks

def load_data_de(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, st_logger):
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
    
    return None

def load_data(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, st_logger):
    
    st_logger.info(f"Presidio text analysis started on the text: {text_with_pii}")
    analyzer = analyzer_engine()
    st_analyze_results = analyze(
        text=text_with_pii,
        language="en",
        score_threshold=0.5,
        allow_list=[],
    )
    st_logger.info(f"Presidio text analysis completed")#: " {st_analyze_results}")

    # Convert each RecognizerResult to a dictionary
    results_as_dicts = [result.to_dict() for result in st_analyze_results]

    # Serialize the list of dictionaries to a JSON string
    results_json = json.dumps(results_as_dicts, indent=2)
    # st_logger.info(f"Presidio text analysis results in JSON: {results_json}")

    st_logger.info(f"Presidio text anonymization started.")
    text_pii_deleted = anonymize(
        text=text_with_pii,
        operator="redact",
        analyze_results=st_analyze_results,
    )
    st_logger.info(f"Text with PII deleted: {text_pii_deleted}")

    text_pii_labeled = anonymize(
        text=text_with_pii,
        operator="replace",
        analyze_results=st_analyze_results,
    )
    st_logger.info(f"Text with PII labeled: {text_pii_labeled}")

    open_ai_params = OpenAIParams(
        openai_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo-instruct",
        api_base=None,
        deployment_id="",
        api_version=None,
        api_type="openai",
    )
    text_chunks = split_text_into_chunks(text_with_pii, max_words=2500)
    text_pii_synthetic_list = []
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
        text_pii_synthetic_list.append(text_chunk_pii_synthetic)
    text_pii_synthetic = ' '.join(text_pii_synthetic_list)
    st_logger.info(f"Synthetic data created: {text_pii_synthetic}")

    # text_pii_dp = diff_privacy_dp_prompt(text_with_pii, epsilon=250)
    text_pii_dp = diff_privacy_diffractor(text_with_pii, epsilon=2)
    st_logger.info(f"Text with PII DP: {text_pii_dp}")

    st_logger.info(f"Started loading the data into Pinecone: {file_name} {file_hash}")
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
        text=text_pii_dp,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_dp"
    )
    st_logger.info(f"Finished loading the data into Pinecone: {file_name} {file_hash}")
    insert_record(table_name, file_name, file_hash, file_bytes, text_with_pii, text_pii_deleted.text, text_pii_labeled.text, text_pii_synthetic, text_pii_dp, results_json)
    st_logger.info("Document inserted into the database.")
    database_file = retrieve_record_by_hash(table_name, file_hash)

    return database_file


def load_data_complex(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, st_logger):
    
    st_logger.info(f"Presidio text analysis started on the text: {text_with_pii}")
    analyzer = analyzer_engine()
    st_analyze_results = analyze(
        text=text_with_pii,
        language="en",
        score_threshold=0.5,
        allow_list=[],
    )
    st_logger.info(f"Presidio text analysis completed")#: " {st_analyze_results}")

    # Convert each RecognizerResult to a dictionary
    results_as_dicts = [result.to_dict() for result in st_analyze_results]

    # Serialize the list of dictionaries to a JSON string
    results_json = json.dumps(results_as_dicts, indent=2)
    # st_logger.info(f"Presidio text analysis results in JSON: {results_json}")

    st_logger.info(f"Presidio text anonymization started.")
    text_pii_deleted = anonymize(
        text=text_with_pii,
        operator="redact",
        analyze_results=st_analyze_results,
    )
    st_logger.info(f"Text with PII deleted: {text_pii_deleted}")

    text_pii_labeled = anonymize(
        text=text_with_pii,
        operator="replace",
        analyze_results=st_analyze_results,
    )
    st_logger.info(f"Text with PII labeled: {text_pii_labeled}")

    open_ai_params = OpenAIParams(
        openai_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo-instruct",
        api_base=None,
        deployment_id="",
        api_version=None,
        api_type="openai",
    )
    text_chunks = split_text_into_chunks(text_with_pii, max_words=2500)
    text_pii_synthetic_list = []
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
        text_pii_synthetic_list.append(text_chunk_pii_synthetic)
    text_pii_synthetic = ' '.join(text_pii_synthetic_list)
    st_logger.info(f"Synthetic data created: {text_pii_synthetic}")

    text_pii_dp_diffractor1 = diff_privacy_diffractor(text_with_pii, epsilon = 1)
    text_pii_dp_diffractor2 = diff_privacy_diffractor(text_with_pii, epsilon = 2)
    text_pii_dp_diffractor3 = diff_privacy_diffractor(text_with_pii, epsilon = 3)

    st_logger.info(f"Started loading the data into Pinecone: {file_name} {file_hash}")
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
        text=text_pii_dp_diffractor1,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_dp_diffractor1"
    )
    loadDataPinecone(
        index_name=index_name,
        text=text_pii_dp_diffractor2,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_dp_diffractor2"
    )   
    loadDataPinecone(
        index_name=index_name,
        text=text_pii_dp_diffractor3,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_dp_diffractor3"
    )
    st_logger.info(f"Finished loading the data into Pinecone: {file_name} {file_hash}")
    insert_record_complex(table_name, file_name, file_hash, file_bytes, text_with_pii, text_pii_deleted.text, text_pii_labeled.text, text_pii_synthetic, text_pii_dp_diffractor1, text_pii_dp_diffractor2, text_pii_dp_diffractor3, results_json)
    st_logger.info("Document inserted into the database.")
    database_file = retrieve_record_by_hash(table_name, file_hash)

    return database_file