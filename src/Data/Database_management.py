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

def add_columns(table_name, columns_dict):
    """
    Add one or more columns to an existing table.
    
    Parameters:
    - table_name: The name of the table to modify
    - columns_dict: A dictionary where keys are column names and values are column types
                   Example: {"new_column": "TEXT", "another_column": "INTEGER"}
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            for column_name, column_type in columns_dict.items():
                cur.execute(f"""
                    ALTER TABLE {table_name} 
                    ADD COLUMN IF NOT EXISTS {column_name} {column_type};
                """)
            conn.commit()
            print(f"Added columns to table '{table_name}' successfully")
    except Exception as e:
        print(f"Error adding columns to table: {e}")
    finally:
        conn.close()

def drop_columns(table_name, columns_list):
    """
    Delete one or more columns from an existing table.
    
    Parameters:
    - table_name: The name of the table to modify
    - columns_dict: A list with all the column names
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            for column_name in columns_list:
                cur.execute(f"""
                    ALTER TABLE {table_name}
                    DROP COLUMN IF EXISTS {column_name};
                """)
            conn.commit()
            print(f"Columns '{columns_list}' from table '{table_name}' dropped successfully")
    except Exception as e:
        print(f"Error dropping column: {e}")
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

def insert_responses(table_name, file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp_diffractor1, response_pii_dp_diffractor2, response_pii_dp_diffractor3, response_pii_dp_dp_prompt1, response_pii_dp_dp_prompt2, response_pii_dp_dp_prompt3, response_pii_dp_dpmlm1, response_pii_dp_dpmlm2, response_pii_dp_dpmlm3, details):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO {table_name} (file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp_diffractor1, response_pii_dp_diffractor2, response_pii_dp_diffractor3, response_pii_dp_dp_prompt1, response_pii_dp_dp_prompt2, response_pii_dp_dp_prompt3, response_pii_dp_dpmlm1, response_pii_dp_dpmlm2, response_pii_dp_dpmlm3, details) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (file_name, question, response_with_pii, response_pii_deleted, response_pii_labeled, response_pii_synthetic, response_pii_dp_diffractor1, response_pii_dp_diffractor2, response_pii_dp_diffractor3, response_pii_dp_dp_prompt1, response_pii_dp_dp_prompt2, response_pii_dp_dp_prompt3, response_pii_dp_dpmlm1, response_pii_dp_dpmlm2, response_pii_dp_dpmlm3, details)
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

def copy_data_to_existing_table(source_table, target_table, columns):
    """
    Copy data from specified columns in a source table to the same columns in an existing target table.
    
    Parameters:
    - source_table: The name of the table to copy data from
    - target_table: The name of the existing table to insert data into
    - columns: A list of column names to copy
    """
    try:
        conn = psycopg2.connect(DATABASE_URL)
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {target_table} ({', '.join(columns)})
                SELECT {', '.join(columns)} FROM {source_table};
            """)
            
            conn.commit()
            print(f"Copied data from columns {columns} in '{source_table}' to '{target_table}' successfully")
    except Exception as e:
        print(f"Error copying data: {e}")
    finally:
        conn.close()

if __name__ == "__main__":

    """
    delete_table("enron_text2")
    create_table_text("enron_text2")
    copy_data_to_existing_table("enron_text", "enron_text2", ["id", "file_name", "file_hash", "pdf_bytes", "text_with_pii", "text_pii_deleted", "text_pii_labeled", "text_pii_synthetic", "details"])

    delete_table("bbc_text2")
    create_table_text("bbc_text2")
    copy_data_to_existing_table("bbc_text", "bbc_text2", ["id", "file_name", "file_hash", "pdf_bytes", "text_with_pii", "text_pii_deleted", "text_pii_labeled", "text_pii_synthetic", "text_pii_dp_diffractor1", "text_pii_dp_diffractor2", "text_pii_dp_diffractor3", "details"]) 

    delete_table("bbc_responses")
    create_table_responses("bbc_responses")

    add_columns("enron_text", {
        "timestamp_modified": "TIMESTAMP", 
        "text_pii_dp_diffractor1": "TEXT",
        "text_pii_dp_diffractor2": "TEXT",
        "text_pii_dp_diffractor3": "TEXT",
        "text_pii_dp_dp_prompt1": "TEXT",
        "text_pii_dp_dp_prompt2": "TEXT",
        "text_pii_dp_dp_prompt3": "TEXT",
        "text_pii_dp_dpmlm1": "TEXT",
        "text_pii_dp_dpmlm2": "TEXT",
        "text_pii_dp_dpmlm3": "TEXT",
    })
    
    drop_columns("enron_text", {
        "timestamp_modified": "TIMESTAMP"
    })
    
    create_table_text("enron_text")
    create_table_text("bbc_text")
    
    create_table_responses("enron_responses")
    create_table_responses("bbc_responses")

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

