import logging
import csv
import hashlib
from Database_management import retrieve_record_by_hash
from Data_loader import load_data, load_data_complex

st_logger = logging.getLogger('enron')
st_logger.setLevel(logging.INFO)
        
def load(table_name, index_name, text_with_pii, file_name, type):
        st_logger.info(f"File name: {file_name}")
        file_bytes = text_with_pii.encode("utf-8")  # Convert the text to bytes
        file_hash = hashlib.sha256(file_bytes).hexdigest()  # Compute hash from bytes
        st_logger.info(f"File hash: {file_hash}")
        if retrieve_record_by_hash(table_name, file_hash) is not None:
            database_file = retrieve_record_by_hash(table_name, file_hash)
            st_logger.info("Existing file found in the database.")
        else:
            st_logger.info("Loading file in the database.")
            if type == 'simple':
                database_file = load_data(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, st_logger)
            elif type == 'complex':
                database_file = load_data_complex(table_name, index_name, text_with_pii, file_name, file_hash, file_bytes, st_logger)
        return database_file
             
def load_enron():
    file_path = "/Users/andreeabodea/ANDREEA/MT/Data/Enron_Stephen_7500_sorted_most_PII.csv"
    index_name = "enron"
    table_name = "enron_text"

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the first row (header)
        for row_number, row in enumerate(csv_reader, start=1):
            if row_number < 92: # Start from ... row
                pass
            if row_number > 100:  # Stop after ... rows
                break
            file_text = row[0]
            file_name = f"Enron_{row_number}"
            load(table_name, index_name, file_text, file_name, 'simple') 

def load_bbc():
    file_path = "/Users/andreeabodea/ANDREEA/MT/Data/BBC_preprocessed.csv"
    index_name = "bbc"
    table_name = "bbc_text"

    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the first row (header)
        for row_number, row in enumerate(csv_reader, start=1):
            if row_number < 3: pass # Start from ... row
            # if row_number > 3: break # Stop after ... rows
            file_text = row[0]
            file_name = f"BBC_{row_number}"
            load(table_name, index_name, file_text, file_name, 'complex') 

if __name__ == "__main__":
    # load_enron()
    load_bbc()
