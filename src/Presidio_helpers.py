"""
Helper methods for the Presidio Streamlit app
"""
from typing import List, Optional, Tuple
import logging
import streamlit as st
from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerResult,
    RecognizerRegistry,
)
from presidio_analyzer.nlp_engine import NlpEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from Presidio_OpenAI import (
    call_completion_model,
    OpenAIParams,
    create_prompt,
)
from Presidio_NLP_engine import (
    create_nlp_engine_with_spacy,
    create_nlp_engine_with_flair
)

logger = logging.getLogger("presidio-streamlit")

@st.cache_resource
def nlp_engine_and_registry() -> Tuple[NlpEngine, RecognizerRegistry]:
    """Create the NLP Engine instance based on the requested model."""

    # "spacy"
    # return create_nlp_engine_with_spacy()
    # "flair" in model_family.lower():
    return create_nlp_engine_with_flair()

    """Create the NLP Engine instance using spaCy."""
    """
    # SPACY
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
        "ner_model_configuration": {
            "model_to_presidio_entity_mapping": {
                "PER": "PERSON",
                "PERSON": "PERSON",
                "NORP": "NRP",
                "FAC": "FACILITY",
                "LOC": "LOCATION",
                "GPE": "LOCATION",
                "LOCATION": "LOCATION",
                "ORG": "ORGANIZATION",
                "ORGANIZATION": "ORGANIZATION",
                "DATE": "DATE_TIME",
                "TIME": "DATE_TIME",
            },
            "low_confidence_score_multiplier": 0.4,
            "low_score_entity_names": ["ORG", "ORGANIZATION"],
        },
    }

    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine)

    return nlp_engine, registry
    """

@st.cache_resource
def analyzer_engine() -> AnalyzerEngine:
    """Create the Analyzer Engine instance."""
    nlp_engine, registry = nlp_engine_and_registry()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)
    return analyzer

@st.cache_resource
def anonymizer_engine():
    """Return AnonymizerEngine."""
    return AnonymizerEngine()

@st.cache_data
def get_supported_entities():
    """Return supported entities from the Analyzer Engine."""
    return analyzer_engine().get_supported_entities() + ["GENERIC_PII"]

@st.cache_data
def analyze(**kwargs):
    """Analyze input using Analyzer engine and input arguments (kwargs)."""
    # The analyzer will use all entities by default when none are specified
    return analyzer_engine().analyze(**kwargs)

def anonymize(
    text: str,
    operator: str,
    analyze_results: List[RecognizerResult],
):
    """Anonymize identified input using Presidio Anonymizer.
    :param text: Full text
    :param operator: Operator name (redact/replace/synthesize/highlight)
    :param analyze_results: list of results from presidio analyzer engine
    """
    if operator == "highlight":
        operator_config = {"lambda": lambda x: x}
        operator = "custom"
    elif operator == "synthesize":
        operator_config = None
        operator = "replace"
    else:
        operator_config = None

    res = anonymizer_engine().anonymize(
        text,
        analyze_results,
        operators={"DEFAULT": OperatorConfig(operator, operator_config)},
    )
    return res

def annotate(text: str, analyze_results: List[RecognizerResult]):
    """Highlight the identified PII entities on the original text
    :param text: Full text
    :param analyze_results: list of results from presidio analyzer engine
    """
    tokens = []
    results = anonymize(
        text=text,
        operator="highlight",
        analyze_results=analyze_results,
    )

    # Sort by start index
    results = sorted(results.items, key=lambda x: x.start)
    for i, res in enumerate(results):
        if i == 0:
            tokens.append(text[: res.start])

        # Append entity text and entity type
        tokens.append((text[res.start : res.end], res.entity_type))

        # If another entity coming i.e. we're not at the last results element, add text up to next entity
        if i != len(results) - 1:
            tokens.append(text[res.end : results[i + 1].start])
        # If no more entities coming, add all remaining text
        else:
            tokens.append(text[res.end :])
    return tokens

def create_fake_data(
    text: str,
    analyze_results: List[RecognizerResult],
    openai_params: OpenAIParams,
):
    """Creates a synthetic version of the text using OpenAI APIs"""
    if not openai_params.openai_key:
        return "Please provide your OpenAI key"
    results = anonymize(text=text, operator="replace", analyze_results=analyze_results)
    prompt = create_prompt(results.text)
    # print(f"Prompt: {prompt}")
    fake = call_completion_model(prompt=prompt, openai_params=openai_params)
    return fake

@st.cache_data
def call_openai_api(
    prompt: str, openai_model_name: str, openai_deployment_name: Optional[str] = None
) -> str:
    """Call OpenAI API with the given prompt and model
    :param prompt: The prompt to send to OpenAI
    :param openai_model_name: The name of the OpenAI model to use
    :param openai_deployment_name: Optional deployment name for Azure OpenAI
    """
    fake_data = call_completion_model(
        prompt, model=openai_model_name, deployment_id=openai_deployment_name
    )
    return fake_data
