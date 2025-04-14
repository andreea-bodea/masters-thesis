from dotenv import load_dotenv
import os
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
                    text_pii_dp_diffractor1 TEXT,
                    text_pii_dp_diffractor2 TEXT,
                    text_pii_dp_diffractor3 TEXT,
                    text_pii_dp_dp_prompt1 TEXT,
                    text_pii_dp_dp_prompt2 TEXT,
                    text_pii_dp_dp_prompt3 TEXT,
                    text_pii_dp_dpmlm1 TEXT,
                    text_pii_dp_dpmlm2 TEXT,
                    text_pii_dp_dpmlm3 TEXT,
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
                    response_pii_dp_diffractor1 TEXT,
                    response_pii_dp_diffractor2 TEXT,
                    response_pii_dp_diffractor3 TEXT,
                    response_pii_dp_dp_prompt1 TEXT,
                    response_pii_dp_dp_prompt2 TEXT,
                    response_pii_dp_dp_prompt3 TEXT,
                    response_pii_dp_dpmlm1 TEXT,
                    response_pii_dp_dpmlm2 TEXT,
                    response_pii_dp_dpmlm3 TEXT,
                    evaluation JSONB
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

def export_table_to_csv(table_name, csv_file_path):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            with open(csv_file_path, 'w', encoding='utf-8') as f:
                cur.copy_expert(f"COPY {table_name} TO STDOUT WITH CSV HEADER", f)
        print(f"Table '{table_name}' exported to '{csv_file_path}' successfully.")
    except Exception as e:
        print(f"Error exporting table '{table_name}': {e}")
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

def insert_record(table_name, file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_diffractor1, text_pii_dp_diffractor2, text_pii_dp_diffractor3, text_pii_dp_dp_prompt1, text_pii_dp_dp_prompt2, text_pii_dp_dp_prompt3, text_pii_dp_dpmlm1, text_pii_dp_dpmlm2, text_pii_dp_dpmlm3, details):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table_name} (file_name, file_hash, pdf_bytes, text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_diffractor1, text_pii_dp_diffractor2, text_pii_dp_diffractor3, text_pii_dp_dp_prompt1, text_pii_dp_dp_prompt2, text_pii_dp_dp_prompt3, text_pii_dp_dpmlm1, text_pii_dp_dpmlm2, text_pii_dp_dpmlm3, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (file_name, file_hash, psycopg2.Binary(pdf_bytes), text_with_pii, text_pii_deleted, text_pii_labeled, text_pii_synthetic, text_pii_dp_diffractor1, text_pii_dp_diffractor2, text_pii_dp_diffractor3, text_pii_dp_dp_prompt1, text_pii_dp_dp_prompt2, text_pii_dp_dp_prompt3, text_pii_dp_dpmlm1, text_pii_dp_dpmlm2, text_pii_dp_dpmlm3, details)
            )
            conn.commit()
            print(f"Record inserted into '{table_name}' successfully")
    except Exception as e:
        print(f"Error inserting record: {e}")
    finally:
        conn.close()

def insert_partial_record(table_name, file_name, file_hash, pdf_bytes, text_with_pii=None, text_pii_deleted=None, text_pii_labeled=None, text_pii_synthetic=None, text_pii_dp_diffractor1=None, text_pii_dp_diffractor2=None, text_pii_dp_diffractor3=None, text_pii_dp_dp_prompt1=None, text_pii_dp_dp_prompt2=None, text_pii_dp_dp_prompt3=None, text_pii_dp_dpmlm1=None, text_pii_dp_dpmlm2=None, text_pii_dp_dpmlm3=None, details=None):
    """
    Insert a record into the specified table with only the provided data.
    
    Parameters:
    - table_name: The name of the table to insert data into
    - Other parameters: Optional data to insert into the table
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            columns = ["file_name", "file_hash", "pdf_bytes"]
            values = [file_name, file_hash, pdf_bytes]
            placeholders = ["%s", "%s", "%s"]

            # Collect only the provided columns and values
            if text_with_pii is not None:
                columns.append("text_with_pii")
                values.append(text_with_pii)
                placeholders.append("%s")
            if text_pii_deleted is not None:
                columns.append("text_pii_deleted")
                values.append(text_pii_deleted)
                placeholders.append("%s")
            if text_pii_labeled is not None:
                columns.append("text_pii_labeled")
                values.append(text_pii_labeled)
                placeholders.append("%s")
            if text_pii_synthetic is not None:
                columns.append("text_pii_synthetic")
                values.append(text_pii_synthetic)
                placeholders.append("%s")
            if text_pii_dp_diffractor1 is not None:
                columns.append("text_pii_dp_diffractor1")
                values.append(text_pii_dp_diffractor1)
                placeholders.append("%s")
            if text_pii_dp_diffractor2 is not None:
                columns.append("text_pii_dp_diffractor2")
                values.append(text_pii_dp_diffractor2)
                placeholders.append("%s")
            if text_pii_dp_diffractor3 is not None:
                columns.append("text_pii_dp_diffractor3")
                values.append(text_pii_dp_diffractor3)
                placeholders.append("%s")
            if text_pii_dp_dp_prompt1 is not None:
                columns.append("text_pii_dp_dp_prompt1")
                values.append(text_pii_dp_dp_prompt1)
                placeholders.append("%s")
            if text_pii_dp_dp_prompt2 is not None:
                columns.append("text_pii_dp_dp_prompt2")
                values.append(text_pii_dp_dp_prompt2)
                placeholders.append("%s")
            if text_pii_dp_dp_prompt3 is not None:
                columns.append("text_pii_dp_dp_prompt3")
                values.append(text_pii_dp_dp_prompt3)
                placeholders.append("%s")
            if text_pii_dp_dpmlm1 is not None:
                columns.append("text_pii_dp_dpmlm1")
                values.append(text_pii_dp_dpmlm1)
                placeholders.append("%s")
            if text_pii_dp_dpmlm2 is not None:
                columns.append("text_pii_dp_dpmlm2")
                values.append(text_pii_dp_dpmlm2)
                placeholders.append("%s")
            if text_pii_dp_dpmlm3 is not None:
                columns.append("text_pii_dp_dpmlm3")
                values.append(text_pii_dp_dpmlm3)
                placeholders.append("%s")
            if details is not None:
                columns.append("details")
                values.append(details)
                placeholders.append("%s")

            # Construct the SQL query dynamically
            cur.execute(
                f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})",
                values
            )
            conn.commit()
            print(f"Partial record inserted into '{table_name}' successfully")
    except Exception as e:
        print(f"Error inserting partial record: {e}")
    finally:
        conn.close()

def insert_responses(table_name, file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp_diffractor1, response_pii_dp_diffractor2, response_pii_dp_diffractor3, response_pii_dp_dp_prompt1, response_pii_dp_dp_prompt2, response_pii_dp_dp_prompt3, response_pii_dp_dpmlm1, response_pii_dp_dpmlm2, response_pii_dp_dpmlm3, evaluation):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table_name} (file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp_diffractor1, response_pii_dp_diffractor2, response_pii_dp_diffractor3, response_pii_dp_dp_prompt1, response_pii_dp_dp_prompt2, response_pii_dp_dp_prompt3, response_pii_dp_dpmlm1, response_pii_dp_dpmlm2, response_pii_dp_dpmlm3, evaluation) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp_diffractor1, response_pii_dp_diffractor2, response_pii_dp_diffractor3, response_pii_dp_dp_prompt1, response_pii_dp_dp_prompt2, response_pii_dp_dp_prompt3, response_pii_dp_dpmlm1, response_pii_dp_dpmlm2, response_pii_dp_dpmlm3, evaluation)
            )
            conn.commit()
            print(f"Response inserted into '{table_name}' successfully")
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

def add_data(table_name, file_hash, **kwargs):
    """
    Update specific columns in an existing record identified by file_hash.
    
    Parameters:
    - table_name: The name of the table
    - file_hash: The hash used to identify the record
    - **kwargs: Column names and values to update
    """
    try:
        if not kwargs:
            print("No columns specified for update")
            return
            
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            # Construct SET clause for the UPDATE statement
            set_clause = ", ".join([f"{key} = %s" for key in kwargs.keys()])
            values = list(kwargs.values())
            values.append(file_hash)  # Add file_hash for the WHERE clause
            
            # Execute the UPDATE statement
            cur.execute(
                f"UPDATE {table_name} SET {set_clause} WHERE file_hash = %s",
                values
            )
            rows_updated = cur.rowcount
            conn.commit()
            
            if rows_updated > 0:
                print(f"Record with file_hash '{file_hash}' updated successfully")
            else:
                print(f"No record found with file_hash '{file_hash}'")
                
    except Exception as e:
        print(f"Error updating record: {e}")
    finally:
        conn.close()

def update_response_evaluation(table_name, file_name, question, evaluation):
    """
    Update the evaluation column for a specific row identified by file_name and question.
    
    Parameters:
    - table_name: The name of the table to update
    - file_name: The name of the file to identify the row
    - question: The question to identify the row
    - evaluation: The JSON string to be stored in the evaluation column
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(f"""
                UPDATE {table_name}
                SET evaluation = %s
                WHERE file_name = %s AND question = %s;
            """, (evaluation, file_name, question))
            conn.commit()
            print(f"Evaluation updated successfully for file: {file_name} and question: {question}")
    except Exception as e:
        print(f"Error updating evaluation: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    """
    create_table_text("enron_text2")
    create_table_text("bbc_text2")
    create_table_responses("enron_responses2")
    create_table_responses("bbc_responses2")
    export_table_to_csv("enron_text2", "./enron_text2.csv")
    export_table_to_csv("bbc_text2", "/./bbc_text2.csv")
    export_table_to_csv("enron_responses2", "/./enron_responses2.csv")
    export_table_to_csv("bbc_responses2", "/./bbc_responses2.csv")
    """