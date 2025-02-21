import os
from dotenv import load_dotenv
import psycopg2

def create_pdf_documents_table():
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents_pdf (
                    id SERIAL PRIMARY KEY,
                    file_name VARCHAR(255) NOT NULL,
                    file_hash VARCHAR(64) NOT NULL UNIQUE,
                    pdf_bytes BYTEA NOT NULL,
                    text_with_pii TEXT,
                    text_pii_deleted TEXT,
                    text_pii_labeled TEXT,
                    text_pii_synthetic TEXT,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, 
                    details TEXT
                );
            """)
            conn.commit()
            print("Table 'pdf_documents' created successfully")
    except Exception as e:
        print(f"Error creating table: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    create_pdf_documents_table() 