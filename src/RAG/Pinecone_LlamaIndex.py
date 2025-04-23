# LLAMAINDEX RAG 
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/PineconeIndexDemo/
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/existing_data/pinecone_existing_data/

# PINECONE VECTORSTORE:
# https://docs.pinecone.io/guides/get-started/quickstart 
# https://docs.pinecone.io/guides/data/upsert-data

import logging 
import os
import dotenv
import openai
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from Data.Database_management import retrieve_record_by_name

logging.basicConfig(level=logging.INFO)

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

openai.api_request_timeout = 30  # Set timeout to 60 seconds

def createOrGetPinecone(index_name: str):
    pinecone = Pinecone(api_key=pinecone_api_key)
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
    document = Document(
        text=text,
        metadata={"file_name": file_name, "file_hash": file_hash, "text_type": text_type}
    )
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents=[document],
        storage_context=storage_context,
        embed_model=embedding_model,
        show_progress=True, 
        chunk_size=2048
    )

def getResponse(index_name: str, question: str, filters: list) -> str:
    pinecone_index = createOrGetPinecone(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store)
    metadata_filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="file_hash", value=filters[0]),
            ExactMatchFilter(key="text_type", value=filters[1])
        ]
    )
    similarity_top_k = 2
    query_engine = index.as_query_engine(llm=llm, streaming=False, similarity_top_k=similarity_top_k, filters=metadata_filters)
    response = query_engine.query(question)

    nodes_text = []
    for i in range( 0, min(similarity_top_k, len(response.source_nodes)) ):
        node_text = response.source_nodes[i].get_text()
        nodes_text.append(node_text)

    evaluation_result = {}
    return (response, nodes_text, evaluation_result)

def load_all(table_name, index_name, file_name_pattern, start, last):
        text_types = ["text_with_pii", "text_pii_deleted", "text_pii_labeled", "text_pii_synthetic", "text_pii_dp_diffractor1", "text_pii_dp_diffractor2", "text_pii_dp_diffractor3", "text_pii_dp_dp_prompt1", "text_pii_dp_dp_prompt2", "text_pii_dp_dp_prompt3", "text_pii_dp_dpmlm1", "text_pii_dp_dpmlm2", "text_pii_dp_dpmlm3"]
        for i in range(start, last+1): # FOR EACH DATABASE FILE
            if table_name == "enron_text2" and i == 61: continue # MISSING TEXT IN ENRON
            file_name = file_name_pattern.format(i)
            database_file = retrieve_record_by_name(table_name, file_name)
            for text_type in text_types:
                loadDataPinecone(
                    index_name=index_name,
                    text=database_file[text_type],
                    file_name=file_name,
                    file_hash=database_file['file_hash'],
                    text_type=text_type
                )
                print(f"Loaded embeddings for: {file_name} {text_type}")

if __name__ == "__main__":
    # load_all(table_name="enron_text2", index_name="enron2", file_name_pattern="Enron_{}", start=1, last=99)
    load_all(table_name="bbc_text2", index_name="bbc2", file_name_pattern="BBC_{}", start=1, last=200)