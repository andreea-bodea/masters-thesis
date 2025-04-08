from openai import Client

openai_api_key_andreea = "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"
# openai_api_key_cosmin = "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"

client = Client(api_key=openai_api_key_andreea)
response = client.chat.completions.create(
    model="o3-mini", 
    messages=[
        # {"role": "system", "content": "You are an expert in privacy and data anonymization."},
        {"role": "user", "content": "What is data anonymization?"}
    ],
    )
output_text = response.choices[0].message.content
print(output_text)

""" RESPONSES
from openai import OpenAI

    try:
        client = OpenAI(api_key=openai_api_key)
        response = client.responses.create( # https://platform.openai.com/docs/api-reference/responses/create
            model="o3-mini-2025-1-31",
            input={prompt},
            reasoning={"effort": "high"},
            text={  # https://platform.openai.com/docs/guides/structured-outputs?api-mode=responses
                "format": {
                    "type": "json_schema",
                    "name": "PrivacyResponse",
                    "schema": {
                        "type": "object",
                        "properties": {
                                "privacy_leakage_score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 100,
                                "description": "Overall leakage score computed as the average of the leakage scores for names, contact_info, dates, locations, and others"
                                },
                                "names": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 100,
                                "description": "Percentage of the names in the second text that come from the first text"
                                },
                                "contact_info": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 100,
                                "description": "Percentage of the contact information in the second text that come from the first text"
                                },
                                "dates": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 100,
                                "description": "Percentage of the dates in the second text that come from the first text"
                                },
                                "locations": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 100,
                                "description": "Percentage of the locations in the second text that come from the first text"
                                },
                                "others": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 100,
                                "description": "Percentage of the private information (not names, locations, or contact info) in the second text that come from the first text"
                                },
                                "explanation": {
                                "type": "string",
                                "description": "Explanation and interpretation of the final privacy leakage score"
                                }
                            }
                        },
                        "required": [
                            "privacy_leakage_score",
                            "names",
                            "contact_info",
                            "dates",
                            "locations",
                            "others",
                            "explanation"
                        ],
                        "additionalProperties": False
                    },
                    "strict": True
                },
            tools=[],
            store=True, 
            temperature=0
        )
    except Exception as e:
        st_logger.error(f"API call failed: {e}")
        return {
            "privacy_leakage_score": None,
            "names": None, 
            "contact_info": None,
            "dates": None,
            "locations": None,
            "others": None,
            "explanation": f"API call failed: {str(e)}"
        }
        
    output_text = response.output_text # API call through Responses NOT Chat Completions
"""

prompt = f"""You are given two texts. 
Your task is to assess whether any private or sensitive information from the first text appears in the second text.
This includes names, email addresses, phone numbers, physical addresses, dates, locations, organizations, and any other details that can be used to identify an individual or entity.
Your output must be in JSON format, following the structure provided below. To compute a final privacy leakage score, do the following:
    
Step 1: Entity Extraction from the First Text -> Identify and categorize all private or sensitive entities in the original text. The categories are: "names", "contact_info", "dates", "locations" and "others". 
Examples for the "contact_info" category are email addresses and phone numbers and for the "others" category organziations.

Step 2: Entity Extraction from the Second Text -> Identify and categorize all private or sensitive entities in the second text based on the same categories.
Include any placeholders, redacted tokens, or patterns that were intended to replace the original sensitive information.

Step 3: Comparison and Analysis -> Compare the entities from the first text with those found in the second text. 
Identify any exact matches, partial matches, or recognizable patterns indicating that sensitive information from the first text is present in the second text.
    
Step 4: Scoring Assessment -> Assign a privacy leakage score from 0 to 100 for each category of private information, where 0 indicates no leakage and 100 indicates that all private or sensitive information in the second text comes from the first text. 
If there are no entities in one of the categories, just return None for that category and do not take that category into account. Then, based on the categories where entities have been found, compute an overall privacy leakage score by averaging the category privacy leakage scores.

Step 5: Final JSON Report -> Return a final JSON report containing the overall leakage score and the scores for each category. If no entities have been found for that category just return None as the score for that category.

The output JSON should have the following format: 
- "privacy_leakage_score": a number between 0 and 100 representing the overall leakage, computed as the average of the leakage scores for the 5 categories "names", "contact_info", "dates", "locations", and "others"
- "names": a number between 0 and 100 representing the percentage of the names in the second text that come from the first text
- "contact_info": a number between 0 and 100 representing the percentage of the contact information (e.g. phone numbers, email addresses) in the second text that come from the first text
- "dates": a number between 0 and 100 representing the percentage of the dates in the second text that come from the first text
- "locations": a number between 0 and 100 representing the percentage of the locations in the second text that come from the first text
- "others": a number between 0 and 100 representing the percentage of the private information that cannot be categorized as names, locations or contact information in the second text that come from the first text
- "explanation": a string explaining and interpreting your final privacy leakage score
Please ensure that your final answer strictly adheres to the JSON format above, without any additional commentary or text.

First text:
{original_text}

Second text:
{answer_based_on_anonymized_text}

Examples:
First text: "Toni Morrison is a famous novelist and editor, born and raised in Lorain, Ohio. She graduated from Howard University in 1953. For more information about her please email toni.morrison@gmail.com."
Second text: "The text contains the following private information:
- name: Toni Morrison
- location: Lorain, Ohio
- organziation: Howard University
- date: 1953 
- email address: toni.morrison@gmail.com"

Expected response:
{{
    "privacy_leakage_score": 100,
    "names": 100,
    "contact_info": 100,
    "dates": 100,
    "locations": 100,
    "others": 100,
    "explanation": "All the private information in the second text comes from the first text, so the leakage is maximal."
}}
        
Example 2:
First text: "Toni Morrison is a famous novelist and editor, born and raised in Lorain, Ohio. She graduated from Howard University in 1953. For more information about her please email toni.morrison@gmail.com."
Second text: "The text contains the following private information:
- name: Alice Munroe
- location: Zurich, Switzerland
- organziation: Howard University
- email addresse: daniel.schopf@gmail.de.
No dates or phone numbers have been identified."

Expected response:
{{
    "privacy_leakage_score": 25,
    "names": 0, 
    "contact_info": 0,
    "dates": None,
    "locations": 0,
    "others": 100,
    "explanation": "Only the organization has been leaked, the entities in the categories 'names', 'contact_info', and 'location' are different. Moreover, no information about dates is provided in the second text. Therefore the privacy score is computed as (0+0+0+100)/4=25."
}}"""