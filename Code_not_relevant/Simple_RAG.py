# https://docs.llamaindex.ai/en/stable/#introduction
# https://docs.llamaindex.ai/en/stable/use_cases/q_and_a/

import os
from dotenv import load_dotenv  
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

documents = SimpleDirectoryReader("../data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What are the types of RAG systems?")
print(response)