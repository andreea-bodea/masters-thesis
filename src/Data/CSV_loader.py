import logging
import csv
import hashlib
from Data.Database_management import retrieve_record_by_hash
from Dataset_preprocessing.Data_loader import load_data_all, load_data_presidio, load_data_diffractor, load_data_dp_prompt, load_data_dpmlm

st_logger = logging.getLogger('enron')
st_logger.setLevel(logging.INFO)
        
def load(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, type, st_logger):
    if type == 'all':
        if retrieve_record_by_hash(table_name, file_hash) is not None:
            database_file = retrieve_record_by_hash(table_name, file_hash)
            st_logger.info("Existing file found in the database.")
        else: 
            database_file = load_data_all(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, st_logger)
    elif type == 'presidio':
        database_file = load_data_diffractor(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, st_logger)
    elif type == 'diffractor':
        database_file = load_data_diffractor(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, st_logger)
    elif type == 'dp_prompt':
        database_file = load_data_dp_prompt(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, st_logger)
    elif type == 'dpmlm':
        database_file = load_data_dpmlm(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, st_logger)
        
    database_file = retrieve_record_by_hash(table_name, file_hash)
    return database_file
             
def load_enron():
    file_path = "/Users/andreeabodea/ANDREEA/MT/Data/Enron_Stephen_7500_sorted_most_PII.csv"
    index_name = "enron"
    table_name = "enron_text2"

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the first row (header)
        for row_number, row in enumerate(csv_reader, start=1):
            # if row_number < 3: pass # Start from ... row
            if row_number > 2: break # Stop after ... rows
            text_with_pii = row[0]
            file_name = f"Enron_{row_number}"
            file_bytes = text_with_pii.encode("utf-8")  # Convert the text to bytes
            file_hash = hashlib.sha256(file_bytes).hexdigest()  # Compute hash from bytes
            st_logger.info(f"File hash: {file_hash}")
            load(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, type='dpmlm')

def load_bbc():
    file_path = "/Users/andreeabodea/ANDREEA/MT/Data/BBC_preprocessed.csv"
    index_name = "bbc"
    table_name = "bbc_text2"

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the first row (header)
        for row_number, row in enumerate(csv_reader, start=1):
            # if row_number < 3: pass # Start from ... row
            if row_number > 2: break # Stop after ... rows
            text_with_pii = row[0]
            file_name = f"BBC_{row_number}"
            file_bytes = text_with_pii.encode("utf-8")  # Convert the text to bytes
            file_hash = hashlib.sha256(file_bytes).hexdigest()  # Compute hash from bytes
            st_logger.info(f"File hash: {file_hash}")
            load(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, type='dpmlm')

if __name__ == "__main__":
    # load_enron()
    load_bbc()
