import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL") # e.g., "postgresql://user:password@localhost/dbname"

def create_table(table_name):
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
                    text_pii_dp_prompt TEXT,
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
    
def insert_record(table_name, file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_prompt, details):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table_name} (file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_prompt, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (file_name, file_hash, psycopg2.Binary(pdf_bytes), text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_prompt, details)
            )
            conn.commit()
    except Exception as e:
        print(f"Error retrieving record by hash: {e}")
    finally:
        conn.close()
    
if __name__ == "__main__":
    create_table("enron")
    # delete_table("pdf_documents")