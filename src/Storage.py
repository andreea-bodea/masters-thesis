import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv() 
DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., "postgresql://user:password@localhost/dbname"
conn = psycopg2.connect(DATABASE_URL)

def check_pdf_exists(file_hash):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM documents_pdf WHERE file_hash = %s", (file_hash,))
        return cur.fetchone()
    
def insert_pdf_record(file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, details):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents_pdf (file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (file_name, file_hash, psycopg2.Binary(pdf_bytes), text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, details)
        )
        conn.commit()

def list_pdf_records():
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, file_name, file_hash, uploaded_at FROM documents_pdf ORDER BY uploaded_at DESC")
        return cur.fetchall()