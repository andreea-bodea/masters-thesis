import os
import dotenv
import logging
from Presidio.Presidio_helpers import analyze, anonymize, create_fake_data, analyzer_engine
from Presidio.Presidio_OpenAI import OpenAIParams

st_logger = logging.getLogger('Presidio ')
st_logger.setLevel(logging.INFO)
dotenv.load_dotenv()

def split_text_into_chunks(text, max_words):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i + max_words])
        chunks.append(chunk) 
    return chunks

def analyze_text_with_presidio(text_with_pii: str, language: str = "en"):
    st_logger.info(f"Presidio text analysis started on the text: {text_with_pii}")
    analyzer = analyzer_engine()
    st_analyze_results = analyze(
        text=text_with_pii,
        language=language,
        score_threshold=0.5,
        allow_list=[],
    )
    st_logger.info(f"Presidio text analysis completed")
    return st_analyze_results

def delete_pii(text_with_pii: str, language: str = "en"):
    st_analyze_results = analyze_text_with_presidio(text_with_pii, language=language)
    st_logger.info(f"Presidio text anonymization started on the text: {text_with_pii}")
    text_pii_deleted = anonymize(
        text=text_with_pii,
        operator="redact",
        analyze_results=st_analyze_results,
    )
    st_logger.info(f"Text with PII deleted: {text_pii_deleted.text}")
    return text_pii_deleted.text

def label_pii(text_with_pii: str, language: str = "en"):
    st_analyze_results = analyze_text_with_presidio(text_with_pii, language=language)
    st_logger.info(f"Presidio text anonymization started on the text: {text_with_pii}")
    text_pii_labeled = anonymize(
        text=text_with_pii,
        operator="replace",
        analyze_results=st_analyze_results,
    )
    st_logger.info(f"Text with PII labeled: {text_pii_labeled.text}")
    return text_pii_labeled.text

def replace_pii(text_with_pii: str, language: str = "en"):
    st_logger.info(f"Presidio text anonymization started on the text: {text_with_pii}")
    open_ai_params = OpenAIParams(
        openai_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo-instruct",
        api_base=None,
        deployment_id="",
        api_version=None,
        api_type="openai",
    )
    text_chunks = split_text_into_chunks(text_with_pii, max_words=2500)
    text_pii_synthetic_list = []
    for chunk in text_chunks:
        st_analyze_chunk_results = analyze_text_with_presidio(chunk, language=language)
        text_chunk_pii_synthetic = create_fake_data(
            chunk,
            st_analyze_chunk_results,
            open_ai_params,
        )
        text_pii_synthetic_list.append(text_chunk_pii_synthetic)
    text_pii_synthetic = ' '.join(text_pii_synthetic_list)
    st_logger.info(f"Synthetic data created: {text_pii_synthetic}")
    return text_pii_synthetic