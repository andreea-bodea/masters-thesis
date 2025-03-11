import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv() 
DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., "postgresql://user:password@localhost/dbname"
conn = psycopg2.connect(DATABASE_URL)

def list_records():
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, file_name, file_hash, uploaded_at FROM documents_2 ORDER BY uploaded_at DESC")
        return cur.fetchall()
    
def retrieve_record_by_name(file_name):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM documents_2 WHERE file_name = %s", (file_name,))
        return cur.fetchone()
    
def retrieve_record_by_hash(file_hash):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM documents_2 WHERE file_hash = %s", (file_hash,))
        return cur.fetchone()
    
def insert_record(file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_prompt, details):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents_2 (file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_prompt, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (file_name, file_hash, psycopg2.Binary(pdf_bytes), text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_prompt, details)
        )
        conn.commit()