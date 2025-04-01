from dotenv import load_dotenv
import os
import ast
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL") # e.g., "postgresql://user:password@localhost/dbname"

def create_table_text(table_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    file_hash VARCHAR(64) NOT NULL UNIQUE,
                    pdf_bytes BYTEA,
                    text_with_pii TEXT,
                    text_pii_deleted TEXT,
                    text_pii_labeled TEXT, 
                    text_pii_synthetic TEXT,
                    text_pii_dp TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                    details TEXT
                );
            """)
            conn.commit()
            print(f"Table '{table_name}' created successfully")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()

def create_table_text_complex(table_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    file_hash VARCHAR(64) NOT NULL UNIQUE,
                    pdf_bytes BYTEA,
                    text_with_pii TEXT,
                    text_pii_deleted TEXT,
                    text_pii_labeled TEXT, 
                    text_pii_synthetic TEXT,
                    text_pii_dp_diffractor1 TEXT,
                    text_pii_dp_diffractor2 TEXT,
                    text_pii_dp_diffractor3 TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                    details TEXT
                );
            """)
            conn.commit()
            print(f"Table '{table_name}' created successfully")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()

def create_table_responses(table_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    question TEXT,
                    response_with_pii TEXT,
                    response_pii_deleted TEXT,
                    response_pii_labeled TEXT, 
                    response_pii_synthetic TEXT,
                    response_pii_dp TEXT,
                    details TEXT
                );
            """)
            conn.commit()
            print(f"Table '{table_name}' created successfully")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()

def create_table_responses_complex(table_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    question TEXT,
                    response_with_pii TEXT,
                    response_pii_deleted TEXT,
                    response_pii_labeled TEXT, 
                    response_pii_synthetic TEXT,
                    response_pii_dp_diffractor1 TEXT,
                    response_pii_dp_diffractor2 TEXT,
                    response_pii_dp_diffractor3 TEXT,
                    details TEXT
                );
            """)
            conn.commit()
            print(f"Table '{table_name}' created successfully")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()

def delete_table(table_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name};")
            conn.commit()
            print(f"Table '{table_name}' deleted successfully")
    except Exception as e:
        print(f"Error deleting table: {e}")
    finally:
        conn.close()

def list_records(table_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT id, file_name, file_hash, uploaded_at FROM {table_name} ORDER BY uploaded_at DESC")
            return cur.fetchall()
    except Exception as e:
        print(f"Error listing the records in table: {e}")
    finally:
        conn.close()
    
def retrieve_record_by_name(table_name, file_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table_name} WHERE file_name = %s", (file_name,))
            return cur.fetchone()
    except Exception as e:
        print(f"Error retrieving record by name: {e}")
    finally:
        conn.close()
    
def retrieve_record_by_hash(table_name, file_hash):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table_name} WHERE file_hash = %s", (file_hash,))
            return cur.fetchone()
    except Exception as e:
        print(f"Error retrieving record by hash: {e}")
    finally:
        conn.close()
    
def insert_record(table_name, file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp, details):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table_name} (file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (file_name, file_hash, psycopg2.Binary(pdf_bytes), text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp, details)
            )
            conn.commit()
    except Exception as e:
        print(f"Error retrieving record by hash: {e}")
    finally:
        conn.close()

def insert_record_complex(table_name, file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_diffractor1, text_pii_dp_diffractor2, text_pii_dp_diffractor3, details):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table_name} (file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_diffractor1, text_pii_dp_diffractor2, text_pii_dp_diffractor3, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (file_name, file_hash, psycopg2.Binary(pdf_bytes), text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_diffractor1, text_pii_dp_diffractor2, text_pii_dp_diffractor3, details)
            )
            conn.commit()
    except Exception as e:
        print(f"Error retrieving record by hash: {e}")
    finally:
        conn.close()
    
def insert_responses(table_name, file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp, details):
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table_name} (file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp, details)
            )
            conn.commit()
    except Exception as e:
        print(f"Error inserting responses: {e}")
    finally:
        conn.close()

def insert_responses_complex(table_name, file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp_diffractor1, response_pii_dp_diffractor2, response_pii_dp_diffractor3, details):
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table_name} (file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp_diffractor1, response_pii_dp_diffractor2, response_pii_dp_diffractor3, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp_diffractor1, response_pii_dp_diffractor2, response_pii_dp_diffractor3, details)
            )
            conn.commit()
    except Exception as e:
        print(f"Error inserting responses: {e}")
    finally:
        conn.close()

def retrieve_responses_by_name(table_name, file_name):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table_name} WHERE file_name = %s", (file_name,))
            return cur.fetchone()
    except Exception as e:
        print(f"Error retrieving record by name: {e}")
    finally:
        conn.close()

def retrieve_responses_by_name_and_question(table_name, file_name, question):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table_name} WHERE file_name = %s AND question = %s", (file_name, question))
            return cur.fetchone()
    except Exception as e:
        print(f"Error retrieving responses by name and question: {e}")
    finally:
        conn.close()

def update_text_pii_dp(table_name, file_name, new_text_pii_dp):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {table_name} SET text_pii_dp = %s WHERE file_name = %s",
                (new_text_pii_dp, file_name)
            )
            conn.commit()
            print(f"Record with file_name '{file_name}' updated successfully")
    except Exception as e:
        print(f"Error updating text_pii_dp: {e}")
    finally:
        conn.close()

def update_text_pii_synthetic(table_name, file_name, new_text_pii_synthetic):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {table_name} SET text_pii_synthetic = %s WHERE file_name = %s",
                (new_text_pii_synthetic, file_name)
            )
            conn.commit()
            print(f"Record with file_name '{file_name}' updated successfully")
    except Exception as e:
        print(f"Error updating text_pii_dp: {e}")
    finally:
        conn.close()

if __name__ == "__main__":

    create_table_responses_complex("bbc_responses")


    """
    create_table_text_complex("bbc_text")


    delete_table("enron_text")
    create_table_text("enron_text")
    
    delete_table("enron_responses")
    create_table_responses("enron_responses")

    database_file = retrieve_record_by_name('enron_text', 'Enron_25')
    print(database_file['text_with_pii'])
    print()
    print(database_file['text_pii_deleted'])
    print()
    print(database_file['text_pii_labeled'])
    print()
    print(database_file['text_pii_synthetic'])   
    print() 
    print(database_file['text_pii_dp'])
    """
    
    """
    # UPDATE text_pii_synthetic 
    ### Enron_61 does NOT exist because it was a duplicate email
    #for nr in range(1, 61):
    for nr in range(62, 92): 
        file_name = f"Enron_{nr}"
        database_file = retrieve_record_by_name('enron_text', file_name)
        text_pii_dp = database_file['text_pii_dp']
        # print(f"File Name: {file_name}")
        # print(f"Text OLD: {text_pii_dp}")
        if text_pii_dp.startswith('[') and text_pii_dp.endswith(']'):
            text_pii_dp_new = ast.literal_eval(text_pii_dp)[0]
            # print(f"Text with PII NEW: {text_pii_dp_new}")
            update_text_pii_dp('enron_text', file_name, text_pii_dp_new)
    """

