# Vectorstores:
# PINECONE https://docs.pinecone.io/guides/get-started/quickstart https://docs.pinecone.io/guides/data/upsert-data

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)

# Create or get a Pinecone index
index_name = "masters-thesis"
if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='gcp',
            region=pinecone_env
        )
    )

pinecone_index = pinecone.Index(index_name)

# Load documents
documents = SimpleDirectoryReader("../data").load_data()

# Set up Pinecone as the vector store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create the index
index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
    show_progress=True
)