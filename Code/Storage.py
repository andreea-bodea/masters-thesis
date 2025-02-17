import os
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL")  # e.g., "postgresql://user:password@localhost/dbname"
conn = psycopg2.connect(DATABASE_URL)

def check_pdf_exists(file_hash):
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT * FROM pdf_documents WHERE file_hash = %s", (file_hash,))
        return cur.fetchone()

def insert_pdf_record(file_name, file_hash, pdf_bytes, extracted_text, anonymized_text):
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO pdf_documents (file_name, file_hash, pdf_bytes, extracted_text, anonymized_text) VALUES (%s, %s, %s, %s, %s)",
            (file_name, file_hash, psycopg2.Binary(pdf_bytes), extracted_text, anonymized_text)
        )
        conn.commit()

def list_pdf_records():
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT id, file_name, uploaded_at FROM pdf_documents ORDER BY uploaded_at DESC")
        return cur.fetchall()