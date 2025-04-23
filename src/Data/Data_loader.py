import json
import logging
from Data.Database_management import insert_partial_record, add_data
from RAG.Pinecone_LlamaIndex import loadDataPinecone
from Presidio.Presidio import analyze_text_with_presidio, delete_pii, label_pii, replace_pii
from Differential_privacy.DP import diff_privacy_dp_prompt, diff_privacy_diffractor, diff_privacy_dpmlm

st_logger = logging.getLogger('Data_loader ')
st_logger.setLevel(logging.INFO)

def load_data_presidio(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes):
    text_pii_deleted = delete_pii(text_with_pii)
    st_logger.info(f"{file_name} - Text with PII deleted: {text_pii_deleted}")
    text_pii_labeled = label_pii(text_with_pii)
    st_logger.info(f"{file_name} - Text with PII labeled: {text_pii_labeled}")
    text_pii_synthetic = replace_pii(text_with_pii)
    st_logger.info(f"{file_name} - Synthetic data created: {text_pii_synthetic}")

    st_analyze_results = analyze_text_with_presidio(text_with_pii)
    results_as_dicts = [result.to_dict() for result in st_analyze_results]
    results_json = json.dumps(results_as_dicts, indent=2)

    insert_partial_record(
        table_name, file_name, file_hash, file_bytes, text_with_pii,
        text_pii_deleted, text_pii_labeled, text_pii_synthetic,
        text_pii_dp_diffractor1=None, text_pii_dp_diffractor2=None, text_pii_dp_diffractor3=None,
        text_pii_dp_dp_prompt1=None, text_pii_dp_dp_prompt2=None, text_pii_dp_dp_prompt3=None,
        text_pii_dp_dpmlm1=None, text_pii_dp_dpmlm2=None, text_pii_dp_dpmlm3=None,
        details=results_json
    )
    st_logger.info(f"Presidio data inserted into the database: {file_name} {file_hash}")

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
    st_logger.info(f"Presidio data inserted into Pinecone: {file_name} {file_hash}")

def load_data_diffractor(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes):
    text_pii_dp_diffractor1 = diff_privacy_diffractor(text_with_pii, epsilon = 1)
    st_logger.info(f"{file_name} - text_pii_dp_diffractor1: {text_pii_dp_diffractor1}")    
    text_pii_dp_diffractor2 = diff_privacy_diffractor(text_with_pii, epsilon = 2)
    st_logger.info(f"{file_name} - text_pii_dp_diffractor2: {text_pii_dp_diffractor2}")    
    text_pii_dp_diffractor3 = diff_privacy_diffractor(text_with_pii, epsilon = 3)
    st_logger.info(f"{file_name} - text_pii_dp_diffractor3: {text_pii_dp_diffractor3}")    
    
    add_data(
        table_name, 
        file_hash, 
        text_pii_dp_diffractor1=text_pii_dp_diffractor1, 
        text_pii_dp_diffractor2=text_pii_dp_diffractor2, 
        text_pii_dp_diffractor3=text_pii_dp_diffractor3
    )
    st_logger.info(f"Diffractor data inserted into the database: {file_name} {file_hash}")

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
    st_logger.info(f"Diffractor data inserted into the database: {file_name} {file_hash}")

def load_data_dp_prompt(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes):
    text_pii_dp_dp_prompt1 = diff_privacy_dp_prompt(text_with_pii, epsilon = 150)
    st_logger.info(f"{file_name} - text_pii_dp_dp_prompt1: {text_pii_dp_dp_prompt1}")
    text_pii_dp_dp_prompt2 = diff_privacy_dp_prompt(text_with_pii, epsilon = 200)
    st_logger.info(f"{file_name} - text_pii_dp_dp_prompt2: {text_pii_dp_dp_prompt2}")
    text_pii_dp_dp_prompt3 = diff_privacy_dp_prompt(text_with_pii, epsilon = 250)
    st_logger.info(f"{file_name} - text_pii_dp_dp_prompt3: {text_pii_dp_dp_prompt3}")

    add_data(
        table_name, 
        file_hash, 
        text_pii_dp_dp_prompt1=text_pii_dp_dp_prompt1, 
        text_pii_dp_dp_prompt2=text_pii_dp_dp_prompt2, 
        text_pii_dp_dp_prompt3=text_pii_dp_dp_prompt3
    )
    st_logger.info(f"DP PROMPT data inserted into the database: {file_name} {file_hash}")

    loadDataPinecone(
        index_name=index_name,
        text=text_pii_dp_dp_prompt1,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_dp_dp_prompt1"
    )
    loadDataPinecone(
        index_name=index_name,
        text=text_pii_dp_dp_prompt2,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_dp_dp_prompt2"
    )   
    loadDataPinecone(
        index_name=index_name,
        text=text_pii_dp_dp_prompt3,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_dp_dp_prompt3"
    )
    st_logger.info(f"DP PROMPT data inserted into Pinecone: {file_name} {file_hash}")

def load_data_dpmlm(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes):
    text_pii_dp_dpmlm1 = diff_privacy_dpmlm(text_with_pii, epsilon = 50)
    st_logger.info(f"{file_name} - text_pii_dp_dpmlm1: {text_pii_dp_dpmlm1}")
    text_pii_dp_dpmlm2 = diff_privacy_dpmlm(text_with_pii, epsilon = 75)
    st_logger.info(f"{file_name} - text_pii_dp_dpmlm2: {text_pii_dp_dpmlm2}")
    text_pii_dp_dpmlm3 = diff_privacy_dpmlm(text_with_pii, epsilon = 100)
    st_logger.info(f"{file_name} - text_pii_dp_dpmlm3: {text_pii_dp_dpmlm3}")

    add_data(
        table_name, 
        file_hash, 
        text_pii_dp_dpmlm1=text_pii_dp_dpmlm1, 
        text_pii_dp_dpmlm2=text_pii_dp_dpmlm2, 
        text_pii_dp_dpmlm3=text_pii_dp_dpmlm3
    )
    st_logger.info(f"DPMLM data inserted into the database: {file_name} {file_hash}")

    loadDataPinecone(
        index_name=index_name,
        text=text_pii_dp_dpmlm1,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_dp_dpmlm1"
    )
    loadDataPinecone(
        index_name=index_name,
        text=text_pii_dp_dpmlm2,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_dp_dpmlm2"
    )   
    loadDataPinecone(
        index_name=index_name,
        text=text_pii_dp_dpmlm3,
        file_name=file_name,
        file_hash=file_hash,
        text_type="text_pii_dp_dpmlm3"
    )
    st_logger.info(f"DPMLM data inserted into Pinecone: {file_name} {file_hash}")

def load_data_all(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, st_logger):

    load_data_presidio(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    load_data_diffractor(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    load_data_dp_prompt(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    load_data_dpmlm(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes)
    st_logger.info("All data inserted into database and Pinecone.")