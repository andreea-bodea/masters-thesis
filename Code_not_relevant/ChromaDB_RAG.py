# Vectorstores:
# CHROMA https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo/
# ChromaDB does NOT provide a web interface to view the collections like Pinecone or Qdrant.

# PINECONE https://docs.llamaindex.ai/en/stable/examples/vector_stores/existing_data/pinecone_existing_data/
# QDRANT https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/
# SUPABASE https://docs.llamaindex.ai/en/stable/examples/vector_stores/SupabaseVectorIndexDemo/

import os
from dotenv import load_dotenv  
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

llm = OpenAI(
    model="gpt-4",
    api_key=openai_api_key,
    max_new_tokens=500,
    temperature=0.0,
)
   
documents = SimpleDirectoryReader("../data").load_data()

"""
# Set up ChromaDB - PERSISTENT CONNECTION
persist_directory = "/path/to/your/chromadb"  # Update this path as needed
db = chromadb.PersistentClient(path=persist_directory)
collection_name = "your-collection-name"  # Update this name as needed
chroma_collection = db.get_or_create_collection(collection_name)
"""

# Set up ChromaDB - Ephemeral CONNECTION
chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents=documents,
    llm=llm,
    storage_context=storage_context,
    show_progress=True
)

query_engine = index.as_query_engine()
response = query_engine.query("What are the types of RAG systems?")
print(response)

"""
print(chroma_client.list_collections()) # Retrieve All Collection Names
print(chroma_collection.get()) # View Documents in Collection
print(chroma_collection.get(include=["metadatas"])) # Check Metadata for Stored Documents
"""