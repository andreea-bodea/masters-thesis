# https://microsoft.github.io/presidio/samples/python/presidio_notebook/
# app on HuggingFace Spaces https://microsoft.github.io/presidio/samples/python/streamlit/

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, OperatorConfig

def anonymize_text(input_text):
    """
    Anonymize sensitive information in a given text using Microsoft Presidio.
    :param input_text: The text to be anonymized (string).
    :return: Tuple containing the original and anonymized text.
    """
    # Initialize Presidio components
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    # Analyze the text to detect sensitive entities
    results = analyzer.analyze(
        text=input_text,
        entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "LOCATION"],
        language="en",
    )

    # Configure how to anonymize each entity
    anonymization_config = {
        "DEFAULT": OperatorConfig("replace", {"new_value": "[ANONYMIZED]"})
    }

    # Anonymize the detected sensitive information
    anonymized_result = anonymizer.anonymize(
        text=input_text,
        analyzer_results=results,
        operators=anonymization_config,
    )

    # Return the original and anonymized text
    return input_text, anonymized_result.text

# Example usage
text_to_anonymize = """
Hello, my name is John Doe. You can contact me at john.doe@example.com or call me at 123-456-7890.
I live in New York City, and my credit card number is 4111-1111-1111-1111.
"""

original_text, anonymized_text = anonymize_text(text_to_anonymize)
# print("Original Text:", original_text)
# print("Anonymized Text:", anonymized_text)
