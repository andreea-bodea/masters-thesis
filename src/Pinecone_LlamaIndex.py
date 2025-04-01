# LLAMAINDEX RAG 
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/PineconeIndexDemo/
# https://docs.llamaindex.ai/en/stable/examples/vector_stores/existing_data/pinecone_existing_data/

# PINECONE VECTORSTORE:
# https://docs.pinecone.io/guides/get-started/quickstart 
# https://docs.pinecone.io/guides/data/upsert-data

# RAG EVALUATION
# https://docs.confident-ai.com/docs/integrations-llamaindex

import logging 
import os
import json
import dotenv
import openai
from pinecone import Pinecone, ServerlessSpec
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from deepeval.integrations.llama_index import (
    DeepEvalAnswerRelevancyEvaluator,
    DeepEvalFaithfulnessEvaluator,
    DeepEvalContextualRelevancyEvaluator,
    DeepEvalBiasEvaluator,
    DeepEvalToxicityEvaluator,
)
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

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
    # evaluation_result = evaluate_response(question, response)

    return (response, nodes_text, evaluation_result)

def evaluate_response(question, response):
    """
    evaluators = {
        "Answer Relevancy": DeepEvalAnswerRelevancyEvaluator(model="gpt-4o-mini-2024-07-18", include_reason=False),
        "Faithfulness": DeepEvalFaithfulnessEvaluator(model="gpt-4o-mini-2024-07-18", include_reason=False),
        "Contextual Relevancy": DeepEvalContextualRelevancyEvaluator(model="gpt-4o-mini-2024-07-18", include_reason=False),
        "Bias": DeepEvalBiasEvaluator(model="gpt-4o-mini-2024-07-18", include_reason=False),
        "Toxicity": DeepEvalToxicityEvaluator(include_reason=False),
    }
    logging.info("Starting evaluation...")
    for name, evaluator in evaluators.items():
        eval_result = evaluator.evaluate_response(query=question, response=response)
        evaluation_result[name] = eval_result.score
    """

    test_case = LLMTestCase(
        input=question,
        actual_output=response.response,
        retrieval_context=[node.get_content() for node in response.source_nodes]
    )

    evaluators = {
        "Answer Relevancy": AnswerRelevancyMetric(model="gpt-4o-mini-2024-07-18", include_reason=False),
        "Faithfulness": FaithfulnessMetric(model="gpt-4o-mini-2024-07-18", include_reason=False),
        "Contextual Relevancy": ContextualRelevancyMetric(model="gpt-4o-mini-2024-07-18", include_reason=False),
       # "Contextual Precision": ContextualPrecisionMetric(model="gpt-4o-mini-2024-07-18", include_reason=False),
       # "Contextual Recall": ContextualRecallMetric(model="gpt-4o-mini-2024-07-18", include_reason=False),
    }

    logging.info("Starting evaluation...")

    for name, metric in evaluators.items():
        metric.measure(test_case)
        evaluation_result[name] = metric.score
