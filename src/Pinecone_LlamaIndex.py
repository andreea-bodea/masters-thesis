# LLAMAINDEX RAG 
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/PineconeIndexDemo/
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/existing_data/pinecone_existing_data/

# PINECONE VECTORSTORE:
# https://docs.pinecone.io/guides/get-started/quickstart 
# https://docs.pinecone.io/guides/data/upsert-data

import logging 
import os
import json
import dotenv
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

logging.basicConfig(level=logging.WARNING)

# Load environment variables
dotenv.load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_region = os.getenv("PINECONE_REGION")
openai_api_key = os.getenv("OPENAI_API_KEY")


# Initialize OpenAI embeddings and LLM
embedding_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=openai_api_key)
llm = OpenAI(
    model="gpt-4o-mini-2024-07-18",
    api_key=openai_api_key,
    max_new_tokens=500,
    temperature=0.0,
)

def createOrGetPinecone(index_name: str):
    # Initialize Pinecone
    pinecone = Pinecone(api_key=pinecone_api_key)

    # Create or get a Pinecone index
    if index_name not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region=pinecone_region
            )
        )
    return pinecone.Index(index_name)

def loadDataPinecone(index_name: str, text: str, file_name: str, file_hash: str, text_type: str):

    pinecone_index = createOrGetPinecone(index_name)

    # Create a document
    document = Document(
        text=text,
        metadata={"file_name": file_name, "file_hash": file_hash, "text_type": text_type}
    )

    # Set up Pinecone as the vector store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create the index and insert the document
    index = VectorStoreIndex.from_documents(
        documents=[document],
        storage_context=storage_context,
        embed_model=embedding_model,
        show_progress=True, 
        chunk_size=2048
    )

def getResponse(index_name: str, question: str, filters: list) -> str:
   
    pinecone_index = createOrGetPinecone(index_name)

    # Set up Pinecone vector store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

    metadata_filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="file_hash", value=filters[0]),
            ExactMatchFilter(key="text_type", value=filters[1])
        ]
    )

    # Create a VectorStoreIndex and query engine
    similarity_top_k = 2
    index = VectorStoreIndex.from_vector_store(vector_store)
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=similarity_top_k, filters=metadata_filters)
    
    # Query the index and return response
    response = query_engine.query(question)

    nodes_text = []
    for i in range( 0, min(similarity_top_k, len(response.source_nodes)) ):
        node_text = response.source_nodes[i].get_text()
        nodes_text.append(node_text)

    return (response, nodes_text)