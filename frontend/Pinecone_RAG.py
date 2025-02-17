# Vectorstores:
# PINECONE 
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/PineconeIndexDemo/
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/existing_data/pinecone_existing_data/

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.response.pprint_utils import pprint_source_node

load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone = Pinecone(api_key=pinecone_api_key)
pinecone_index = pinecone.Index("masters-thesis")

# Set up Pinecone vector store
vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index, text_key="content"
)

# Initialize OpenAI LLM
llm = OpenAI(
    model="gpt-4o-mini-2024-07-18",
    api_key=openai_api_key,
    max_new_tokens=500,
    temperature=0.0,
)

# Create a VectorStoreIndex and query engine
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(llm=llm)

# Query the index
response = query_engine.query("What are the types of RAG systems?")
print(response)
print()

# Print the source node for debugging
nodes = index.as_retriever(similarity_top_k=1).retrieve("What are the types of RAG?")
pprint_source_node(nodes[0])

