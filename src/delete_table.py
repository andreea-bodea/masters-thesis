import os
from dotenv import load_dotenv
import psycopg2

def delete_documents_table():
    load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL")
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS documents;")
            conn.commit()
            print("Table 'documents' deleted successfully")
    except Exception as e:
        print(f"Error deleting table: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    delete_documents_table()