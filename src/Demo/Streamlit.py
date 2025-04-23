# pip install -e .
# streamlit run Streamlit.py

import logging
import pandas as pd
import streamlit as st
import json
import asyncio
import plotly.express as px

# Set logging level to ERROR to reduce noisy logs
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)

from Demo.PDF_reader import convert_pdf_to_text
from Presidio.Presidio import analyze_text_with_presidio, delete_pii, label_pii, replace_pii
from Presidio.Presidio_helpers import annotate
from annotated_text import annotated_text
from Differential_privacy.DP import diff_privacy_diffractor, diff_privacy_dp_prompt, diff_privacy_dpmlm
from RAG.Response_evaluation import calculate_rouge1, calculate_rougeL, calculate_bleu, calculate_cosine_similarity, calculate_perplexity, calculate_privacy_llm_judge, extract_llm_score
from RAG.Local_LlamaIndex import get_offline_RAG_response

try:
    st.set_page_config(
        page_title="GuardRAG",
        layout="wide", #"centered" 
        initial_sidebar_state="collapsed" #"expanded",
    )
except Exception as e:
    pass 

st.title("GuardRAG LIVE 🔍")
st.subheader("Protecting private data in retrieval-augmented generation systems.")

# About section with information about the app
with st.expander("**‼️ About GuardRAG**", expanded=False):
    st.markdown("""
    ### About GuardRAG
    
    GuardRAG is a real-time Retrieval-Augmented Generation (RAG) system, 
    that demonstrates how to protect sensitive information in RAG systems using 
    various anonymization techniques and differential privacy-based methods  
    while preserving the utility of generated responses.
    
    #### Privacy-preserving Methods
        
    **A. Anonymisation**:
    
    **PII Deletion**  
    - Detects PII entities from the input text using NER models (spaCy or FLAIR) via Microsoft Presidio  
    - Completely removes all detected entities   
    - ✅ **Advantages**: Maximizes privacy by eliminating sensitive content entirely  
    - ⚠️ **Disadvantages**: Can reduce coherence and remove useful context

    **PII Labeling**  
    - Detects PII entities from the input text using NER models (spaCy or FLAIR) via Microsoft Presidio  
    - Replaces entities with generic labels such as `<PERSON>`, `<LOCATION>`, `<DATE_TIME>`  
    - ✅ **Advantages**: Maintains sentence structure and grammar; reversible if mappings are saved  
    - ⚠️ **Disadvantages**: Generic labels may reduce readability and interpretability

    **PII Replacement with Synthetic Data**  
    - Detects PII entities from the input text using NER models (spaCy or FLAIR) via Microsoft Presidio  
    - Replaces detected entities with placeholders (e.g., `<PERSON>`, `<DATE>`)  
    - Uses OpenAI's `gpt-3.5-turbo-instruct` to generate realistic, synthetic replacements  
    - ✅ **Advantages**: Retains fluency and realism; preserves structure while protecting privacy  
    - ⚠️ **Disadvantages**: More computationally expensive and reliant on the generative model's quality
                
    **B. Paraphrasing based on Differential Privacy (DP)**: 

    **1-Diffractor**
    - Tokenizes the input text 
    - Classifies the tokens and ignores stopwords (e.g. "the", "and") and punctuation
    - For each remaining token, creates embeddings from pre-trained models like GloVe or Word2Vec
    - Uses FAISS to finds the top N semantically similar tokens (neighbors)
    - Instead of choosing the closest token, a **geometric DP mechanism** selects one at random, with higher chance for similar words
    - Replaces the original token with the chosen 
    - 🔑 **ε (epsilon)**: Controls privacy vs. accuracy. Lower ε → more privacy, noisier text. Larger ε → clearer text, but less privacy. Try ε = 1 or 2 or 3.
    - ✅ **Advantages**: Protects against re-identification by making words fuzzy, not just redacted; Preserves general meaning better than deletion
    - ⚠️ **Disadvantages**: Depending on the chosen epsilon, it may change the meaning of the sentence; too much randomness may hurt clarity of the sentence
    
    **DP-Prompt**: 
    - Tokenizes the input text using a transformer tokenizer
    - Formats the input like into "Document: [original text]. Paraphrase of the document: "
    - Clipps the model's prediction scores (logits) to limit their range (Logit Clipping)
    - Calculates the sensitivity (how much influence one token can have) using the clipped range
    - Calculates the temperature ensuring differential privacy: temperature = (2 * sensitivity) / ε
    - Generates a paraphrased version of the text through the transformer model using **temperature-controlled sampling**
    - 🔑 **ε (epsilon)**: Controls privacy vs. accuracy. Lower ε → more privacy, noisier text. Larger ε → clearer text, but less privacy. Try ε = 150 or 200 or 250. 
    - ✅ **Advantages**: Produces full, fluent paraphrases instead of word-by-word changes; Works well at sentence-level.
    - ⚠️ **Disadvantages**: Too low ε results in vague, truncated, or confusing outputs; Not great for **short texts** or word-level use.

    **DP-MLM (Differentially Private Masked Language Model)**
    - Masks each word in the sentence one at a time
    - Pairs the masked sentence with the original version to give context
    - Uses a masked language model to predicts the hidden word
    - Clips the output logits to limit sensitivity
    - Calculates the temperature ensuring differential privacy: temperature = (2 * sensitivity) / ε
    - Samples the prediction using the temperature 
    - Replaces the masked word with the predicted word
    - 🔑 **ε (epsilon)**: Controls privacy vs. accuracy. Lower ε → more privacy, noisier text. Larger ε → clearer text, but less privacy. Try ε = 50 or 75 or 100.
    - ✅ **Advantages**: Preserves sentence structure and meaning better than simple word substitution; High flexibility for granular anonymization.
    - ⚠️ **Disadvantages**: Too short inputs result in low-quality predictions; Outputs can lose fluency if context is poor or ε is very low.                
    
    #### Evaluation Methods
        
    **Utility Metrics**:
    - **Rouge-1**: Measures unigram overlap (recall-focused)
    - **Rouge-L**: Evaluates longest common subsequence similarity
    - **BLEU**: Calculates n-gram precision compared to a reference
    - **Cosine Similarity**: Measures semantic closeness between response embeddings
    - **Perplexity**: Evaluates language fluency (lower values are better)
    
    **Privacy Metric**:
    - **LLM-as-a-Judge**: Uses an LLM to determine the percentage of privacy leakage in the generated response
    
    The evaluation process helps determine the balance between maintaining response utility while protecting sensitive information.
    """)

# German language support info in a separate expander
with st.expander("**🇩🇪 German Language Support**", expanded=False):
    st.markdown("""
    ### German Language Support
    
    GuardRAG offers support for German text inputs across its privacy-preserving methods:
    
    #### PII-based Methods for German
    - **PII Deletion, Labeling, and Replacement** use Microsoft Presidio with the German NER model `de_core_news_lg` for high-quality entity recognition.
    
    #### Differential Privacy Methods for German
    - **Diffractor**: While functional with German texts, it uses English-based word embeddings which may reduce effectiveness for uniquely German terms.
    - **DP-Prompt**: Uses the multilingual mT5 model (`google/mt5-base`) which provides good support for German texts.
    - **DP-MLM**: Uses a specialized German BERT model (`dbmdz/bert-base-german-cased`) for optimal performance with German text.
    
    #### PDF Processing for German
    - The PDF reader uses language-specific prompts to extract German text accurately.
    
    #### Recommended Epsilon Values
    - The recommended epsilon values are calibrated for both English and German texts.
    - For sensitive German documents, consider using lower epsilon values for greater privacy.
    """)

st_logger = logging.getLogger('streamlit')
st_logger.setLevel(logging.INFO)

# Initialize session state for persistent storage
if 'text_with_pii' not in st.session_state:
    st.session_state.text_with_pii = None
if 'text_anonymized' not in st.session_state:
    st.session_state.text_anonymized = None
if 'selected_anonymization' not in st.session_state:
    st.session_state.selected_anonymization = None
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"

uploaded_file = None
text_with_pii = None
text_anonymized = None
selected_anonymization = None
selected_epsilon = None
response_original = None
response_anonymized = None

### DATA SELECTION ###

st.markdown("---")
st.subheader("📝 ANONYMISATION / PARAPHRASING : upload a file or type in a text and choose the privacy-preserving method")

# Initialize lang_code with default value
lang_code = "en"

col1, col2, col3 = st.columns([1, 1, 1])

# INPUT - COLUMN 1: UPLOADED FILE
uploaded_file = col1.file_uploader("Upload a file:", type=["pdf", "txt"])

# Add example text options to column 1
with col1.expander("📋 Load example text"):
    example_texts = {
        "English - Ministry Document": """MINISTRY OF ECONOMIC AFFAIRS
INTERNAL MEMORANDUM
CONFIDENTIAL

Date: November 14, 2023
Ref: ECO/2023/11/452

To: Department Heads
From: Minister James Wilson
Subject: Q4 Budget Allocation and Upcoming Reforms

Following our meeting on November 10, 2023 at 10:00 AM at the Wellington Conference Room, I am pleased to announce the allocation of £25 million additional funding for the Regional Development Initiative. Director Elizabeth Parker (elizabeth.parker@gov.uk, 020 7946 8321) will oversee distribution to the prioritized regions, particularly focusing on the Manchester and Birmingham economic zones.

Deputy Minister Robert Thompson has scheduled individual consultations with stakeholders from December 5-15 at our London headquarters (125 Victoria Street, London SW1E 6DE). Department heads should contact my executive assistant, Jane Cooper (jane.cooper@gov.uk), to confirm attendance.

The proposed Manufacturing Sector Support Act will be presented to Parliament on January 12, 2024. Our legal team, led by Christopher Davis, Esq., has prepared the necessary documentation. The confidential draft can be accessed via the secure server using your government ID credentials.

Please note: The upcoming strategy meeting with Prime Minister Johnson has been rescheduled to December 3, 2023 at 9:30 AM.

Bank Details for Project Funding:
Account Name: UK Treasury Economic Development
Account Number: 73924685
Sort Code: 20-41-57
IBAN: GB29 BARC 2041 5773 9246 85

Regards,
James Wilson
Minister of Economic Affairs
Direct line: 020 7946 8300
Mobile: +44 7700 900123
Email: james.wilson@cabinet-office.gov.uk
Passport Number: 548973215
National Insurance Number: AB123456C""",
        "German - Ministry Document": """BUNDESMINISTERIUM FÜR WIRTSCHAFT UND ENERGIE
INTERNE MITTEILUNG
VERTRAULICH

Datum: 14. November 2023
Aktenzeichen: BMWI/2023/11/452

An: Abteilungsleiter
Von: Minister Dr. Thomas Schmidt
Betreff: Haushaltsplanung für Q4 und anstehende Reformen

Nach unserer Sitzung am 10. November 2023 um 10:00 Uhr im Konferenzraum Adenauer, freue ich mich, die Bereitstellung von zusätzlichen 22 Millionen Euro für die Regionale Entwicklungsinitiative bekanntzugeben. Direktorin Dr. Sabine Müller (sabine.mueller@bmwi.bund.de, 030 18615 7423) wird die Verteilung an die priorisierten Regionen überwachen, mit besonderem Fokus auf die Wirtschaftszonen in München und Hamburg.

Staatssekretär Dr. Michael Weber hat vom 5. bis 15. Dezember Einzelkonsultationen mit Interessenvertretern in unserem Berliner Hauptsitz (Scharnhorststraße 34-37, 10115 Berlin) angesetzt. Abteilungsleiter sollten sich mit meiner persönlichen Referentin, Frau Julia Fischer (julia.fischer@bmwi.bund.de), in Verbindung setzen, um ihre Teilnahme zu bestätigen.

Der Entwurf des Industrieförderungsgesetzes wird am 12. Januar 2024 dem Bundestag vorgelegt. Unser Rechtsteam unter der Leitung von Rechtsanwalt Dr. Andreas Becker hat die erforderlichen Unterlagen vorbereitet. Auf den vertraulichen Entwurf kann über den sicheren Server mit Ihren Regierungsausweisdaten zugegriffen werden.

Bitte beachten Sie: Das anstehende Strategietreffen mit Bundeskanzler Scholz wurde auf den 3. Dezember 2023 um 9:30 Uhr verschoben.

Bankverbindung für Projektfinanzierung:
Kontoinhaber: Bundeshaushalt Wirtschaftsförderung
Kontonummer: 7392468500
BLZ: 10000000
IBAN: DE89 3704 0044 0532 0130 00
BIC: COBADEFFXXX

Mit freundlichen Grüßen,
Dr. Thomas Schmidt
Bundesminister für Wirtschaft und Energie
Direktwahl: 030 18615 7400
Mobiltelefon: +49 170 1234567
E-Mail: thomas.schmidt@bmwi.bund.de
Personalausweisnummer: L01X34R82
Steuer-ID: 47 123 456 789""",
        "English - Contract": """SERVICES AGREEMENT

CONTRACT No.: SA-2023-4572
Date: February 15, 2023

BETWEEN:

NORTHBRIDGE CONSULTING LTD. (hereinafter referred to as the "Company")
Company Registration No.: 08745621
Registered Office: 45 Blackfriars Road, London, SE1 8NZ, United Kingdom
Represented by: Richard Thompson, Chief Executive Officer
Email: r.thompson@northbridge-consulting.co.uk
Phone: +44 20 7123 4567

AND:

EASTWOOD TECHNOLOGIES INC. (hereinafter referred to as the "Contractor")
Company Registration No.: US-543219876
Registered Office: 789 Tech Parkway, Suite 300, Boston, MA 02110, United States
Represented by: Jennifer Wilson, Director of Operations
Email: jennifer.wilson@eastwood-tech.com
Phone: +1 617 555 8901
Tax ID: 82-4731509

TERMS AND CONDITIONS:

1. SERVICES
The Contractor shall provide software development services as specified in Annex A for the Company's Project Falcon (Project ID: PRJ-2023-0472).

2. TERM
This Agreement shall commence on March 1, 2023 and shall continue until February 28, 2024, unless terminated earlier.

3. PAYMENT
3.1 The Company shall pay the Contractor a fixed fee of £75,000 (seventy-five thousand pounds sterling) for the Services.
3.2 Payment schedule:
    - 30% upon signing this Agreement
    - 40% upon delivery of Phase 1 (due June 30, 2023)
    - 30% upon final delivery (due December 15, 2023)
3.3 Payment details:
    Bank: First National Bank
    Account Name: Eastwood Technologies Inc.
    Account Number: 456789123
    Sort Code: 11-22-33
    SWIFT/BIC: FNBAUS33
    IBAN: US45 FNBA 1234 5678 9012 34

4. CONFIDENTIALITY
All information exchanged between the parties shall be considered confidential and shall not be disclosed to third parties.

5. INTELLECTUAL PROPERTY
All intellectual property created under this Agreement shall be the exclusive property of the Company.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.

For and on behalf of NORTHBRIDGE CONSULTING LTD.:
Signature: _______________________________
Name: Richard Thompson
Position: Chief Executive Officer
Date: February 15, 2023
Passport Number: GBR957284613

For and on behalf of EASTWOOD TECHNOLOGIES INC.:
Signature: _______________________________
Name: Jennifer Wilson
Position: Director of Operations
Date: February 15, 2023
Driver's License: MA982745619""",
        "German - Contract": """DIENSTLEISTUNGSVERTRAG

VERTRAG Nr.: DL-2023-8754
Datum: 15. Februar 2023

ZWISCHEN:

MÜLLER & SCHMIDT GMBH (nachfolgend "Auftraggeber" genannt)
Handelsregisternummer: HRB 98765 (Amtsgericht München)
Geschäftssitz: Rosenheimer Straße 143, 81671 München, Deutschland
Vertreten durch: Dr. Klaus Müller, Geschäftsführer
E-Mail: k.mueller@mueller-schmidt.de
Telefon: +49 89 12345678
USt-IdNr.: DE987654321

UND:

WEBER TECHNOLOGIE AG (nachfolgend "Auftragnehmer" genannt)
Handelsregisternummer: CHE-123.456.789 (Handelsregister Zürich)
Geschäftssitz: Bahnhofstrasse 42, 8001 Zürich, Schweiz
Vertreten durch: Sabine Weber, Vorstandsvorsitzende
E-Mail: s.weber@weber-technologie.ch
Telefon: +41 44 987 6543
USt-IdNr.: CHE-123.456.789 MWST

VERTRAGSBEDINGUNGEN:

1. LEISTUNGEN
Der Auftragnehmer erbringt IT-Beratungsleistungen gemäß Anhang A für das Projekt "Digitale Transformation" (Projekt-ID: PRJ-2023-0891) des Auftraggebers.

2. VERTRAGSLAUFZEIT
Dieser Vertrag beginnt am 1. März 2023 und endet am 28. Februar 2024, sofern er nicht vorzeitig gekündigt wird.

3. VERGÜTUNG
3.1 Der Auftraggeber zahlt dem Auftragnehmer ein festes Honorar von 85.000 € (fünfundachtzigtausend Euro) für die Leistungen.
3.2 Zahlungsplan:
    - 30% bei Vertragsunterzeichnung
    - 40% bei Ablieferung der Phase 1 (fällig am 30. Juni 2023)
    - 30% bei endgültiger Abnahme (fällig am 15. Dezember 2023)
3.3 Zahlungsdetails:
    Bank: Deutsche Bank
    Kontoinhaber: Weber Technologie AG
    Kontonummer: 123456789
    BLZ: 70070010
    SWIFT/BIC: DEUTDEMMXXX
    IBAN: DE89 7007 0010 0123 4567 89

4. VERTRAULICHKEIT
Alle zwischen den Parteien ausgetauschten Informationen gelten als vertraulich und dürfen nicht an Dritte weitergegeben werden.

5. GEISTIGES EIGENTUM
Alle im Rahmen dieses Vertrags geschaffenen geistigen Eigentumsrechte sind ausschließliches Eigentum des Auftraggebers.

ZU URKUND DESSEN haben die Parteien diesen Vertrag zum eingangs genannten Datum unterzeichnet.

Für und im Namen der MÜLLER & SCHMIDT GMBH:
Unterschrift: _______________________________
Name: Dr. Klaus Müller
Position: Geschäftsführer
Datum: 15. Februar 2023
Personalausweisnummer: L22CK47D9

Für und im Namen der WEBER TECHNOLOGIE AG:
Unterschrift: _______________________________
Name: Sabine Weber
Position: Vorstandsvorsitzende
Datum: 15. Februar 2023
Passnummer: C5472D88L"""
    }
    
    ex_col1, ex_col2 = st.columns(2)
    
    with ex_col1:
        if st.button("Load English Ministry Example"):
            st.session_state['example_text'] = example_texts["English - Ministry Document"]
            # Also update the language selection
            st.session_state['selected_language'] = "English"
            st.rerun()
            
        if st.button("Load English Contract Example"):
            st.session_state['example_text'] = example_texts["English - Contract"]
            st.session_state['selected_language'] = "English"
            st.rerun()
            
    with ex_col2:
        if st.button("Load German Ministry Example"):
            st.session_state['example_text'] = example_texts["German - Ministry Document"]
            # Also update the language selection
            st.session_state['selected_language'] = "German"
            st.rerun()
            
        if st.button("Load German Contract Example"):
            st.session_state['example_text'] = example_texts["German - Contract"]
            st.session_state['selected_language'] = "German"
            st.rerun()

if uploaded_file is not None:
    with st.spinner('Reading text...'):
        try:
            file_extension = uploaded_file.name.split('.')[-1]
            file_bytes = uploaded_file.read()
            st_logger.info(f"File uploaded: {uploaded_file.name}, size: {len(file_bytes)} bytes, type: {file_extension}")
            
            if file_extension == "pdf":
                st_logger.info("Processing PDF file...")
                # Use current lang_code (will be default "en" or previously selected language)
                text_with_pii = asyncio.run(convert_pdf_to_text(file_bytes, language=lang_code))
                st_logger.info(f"PDF text extracted, length: {len(text_with_pii) if text_with_pii else 0} characters")
                if not text_with_pii or len(text_with_pii.strip()) == 0:
                    st.warning("The PDF couldn't be read or contains no text. Please try another file.")
            elif file_extension == "txt":
                text_with_pii = file_bytes.decode("utf-8")
                st_logger.info(f"TXT file read, length: {len(text_with_pii)} characters")
            
            # Reset file position for future reads
            uploaded_file.seek(0)
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st_logger.error(f"Error reading file: {str(e)}", exc_info=True)
            text_with_pii = None

# INPUT - COLUMN 2: TYPED TEXT
with col2:
    # Initialize the example text state if it doesn't exist
    if 'example_text' not in st.session_state:
        st.session_state['example_text'] = ""
    
    # Use the example text if it exists in session state
    initial_text = st.session_state['example_text'] if st.session_state['example_text'] else ""
    typed_text = st.text_area("Or type in a text:", value=initial_text, height=200)
    send_button_clicked = st.button("Send")
    
    # Only use typed text if it's not empty and there's no uploaded file text
    if typed_text and (not text_with_pii or len(text_with_pii.strip()) == 0):
        text_with_pii = typed_text
        st_logger.info(f"Using typed text, length: {len(text_with_pii)} characters")

# INPUT - COLUMN 3 
language_map = {"English": "en", "German": "de"}

# Use the session state for language selection if it exists
default_language_index = 0  # Default to English
if 'selected_language' in st.session_state:
    # Find the index of the language in the options list
    try:
        default_language_index = list(language_map.keys()).index(st.session_state['selected_language'])
    except ValueError:
        default_language_index = 0  # Default to English if not found

selected_language = col3.radio(
    "Select text language:",
    options=list(language_map.keys()),
    index=default_language_index,
)
lang_code = language_map[selected_language]
st.session_state.selected_language = selected_language

anonymization_types = [
    "PII Deletion",
    "PII Labeling",
    "PII Replacement with Synthetic Data",
    "Diffractor", 
    "DP-Prompt", 
    "DP-MLM"
]
anonymization_type_map = {
    "PII Deletion": "text_pii_deleted",
    "PII Labeling": "text_pii_labeled",
    "PII Replacement with Synthetic Data": "text_pii_synthetic",
    "Diffractor": "text_pii_dp_diffractor",
    "DP-Prompt": "text_pii_dp_dp_prompt",
    "DP-MLM": "text_pii_dp_dpmlm"
}

mitigations_explanation = """
**PII Deletion**  
- Detects PII entities from the input text using NER models (spaCy or FLAIR) via Microsoft Presidio  
- Completely removes all detected entities   
- ✅ **Advantages**: Maximizes privacy by eliminating sensitive content entirely  
- ⚠️ **Disadvantages**: Can reduce coherence and remove useful context

**PII Labeling**  
- Detects PII entities from the input text using NER models (spaCy or FLAIR) via Microsoft Presidio  
- Replaces entities with generic labels such as `<PERSON>`, `<LOCATION>`, `<DATE_TIME>`  
- ✅ **Advantages**: Maintains sentence structure and grammar; reversible if mappings are saved  
- ⚠️ **Disadvantages**: Generic labels may reduce readability and interpretability

**PII Replacement with Synthetic Data**  
- Detects PII entities from the input text using NER models (spaCy or FLAIR) via Microsoft Presidio  
- Replaces detected entities with placeholders (e.g., `<PERSON>`, `<DATE>`)  
- Uses OpenAI's `gpt-3.5-turbo-instruct` to generate realistic, synthetic replacements  
- ✅ **Advantages**: Retains fluency and realism; preserves structure while protecting privacy  
- ⚠️ **Disadvantages**: More computationally expensive and reliant on the generative model's quality

**1-Diffractor**
- Tokenizes the input text 
- Classifies the tokens and ignores stopwords (e.g. "the", "and") and punctuation
- For each remaining token, creates embeddings from pre-trained models like GloVe or Word2Vec
- Uses FAISS to finds the top N semantically similar tokens (neighbors)
- Instead of choosing the closest token, a **geometric DP mechanism** selects one at random, with higher chance for similar words
- Replaces the original token with the chosen 
- 🔑 **ε (epsilon)**: Controls privacy vs. accuracy. Lower ε → more privacy, noisier text. Larger ε → clearer text, but less privacy. Try ε = 1 or 2 or 3.
- ✅ **Advantages**: Protects against re-identification by making words fuzzy, not just redacted; Preserves general meaning better than deletion
- ⚠️ **Disadvantages**: Depending on the chosen epsilon, it may change the meaning of the sentence; too much randomness may hurt clarity of the sentence
 
**DP-Prompt**: 
- Tokenizes the input text using a transformer tokenizer
- Formats the input like into "Document: [original text]. Paraphrase of the document: "
- Clipps the model's prediction scores (logits) to limit their range (Logit Clipping)
- Calculates the sensitivity (how much influence one token can have) using the clipped range
- Calculates the temperature ensuring differential privacy: temperature = (2 * sensitivity) / ε
- Generates a paraphrased version of the text through the transformer model using **temperature-controlled sampling**
- 🔑 **ε (epsilon)**: Controls privacy vs. accuracy. Lower ε → more privacy, noisier text. Larger ε → clearer text, but less privacy. Try ε = 150 or 200 or 250.
- ✅ **Advantages**: Produces full, fluent paraphrases instead of word-by-word changes; Works well at sentence-level.
- ⚠️ **Disadvantages**: Too low ε results in vague, truncated, or confusing outputs; Not great for **short texts** or word-level use.

**DP-MLM (Differentially Private Masked Language Model)**
- Masks each word in the sentence one at a time
- Pairs the masked sentence with the original version to give context
- Uses a masked language model to predicts the hidden word
- Clips the output logits to limit sensitivity
- Calculates the temperature ensuring differential privacy: temperature = (2 * sensitivity) / ε
- Samples the prediction using the temperature 
- Replaces the masked word with the predicted word
- 🔑 **ε (epsilon)**: Controls privacy vs. accuracy. Lower ε → more privacy, noisier text. Larger ε → clearer text, but less privacy. Try ε = 50 or 75 or 100.
- ✅ **Advantages**: Preserves sentence structure and meaning better than simple word substitution; High flexibility for granular anonymization.
- ⚠️ **Disadvantages**: Too short inputs result in low-quality predictions; Outputs can lose fluency if context is poor or ε is very low.
"""
selected_anonymization = col3.selectbox(
    "Select privacy-preserving method:", 
    anonymization_types, 
    help=mitigations_explanation,
    label_visibility="visible",
    index=0
)

# Recommended epsilon values based on method and language
epsilon_recommendations = {
    "Diffractor": {"en": (1, 10), "de": (1, 10)},
    "DP-Prompt": {"en": (0, 300), "de": (0, 300)},
    "DP-MLM": {"en": (0, 300), "de": (0, 300)}
}

# Customize the epsilon slider based on selected method
if selected_anonymization in ["Diffractor", "DP-Prompt", "DP-MLM"]:
    min_value, max_value = epsilon_recommendations.get(selected_anonymization, {}).get(lang_code, (0, 300))
    default_value = (min_value + max_value) // 2
    
    epsilon_help = f"Recommended epsilon range for {selected_anonymization} in {selected_language}: {min_value}-{max_value}. Lower values provide more privacy but less utility."
    selected_epsilon = col3.slider(
        label="Epsilon value",
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        help=epsilon_help,
    )
else:
    # Hide epsilon slider for non-DP methods
    selected_epsilon = 0

# Language compatibility warnings for each method
if selected_language == "German":
    if selected_anonymization == "Diffractor":
        col3.info("⚠️ Diffractor for German texts may have reduced accuracy as it uses English embedding models. Results may vary.")
    elif selected_anonymization == "DP-Prompt":
        col3.success("✅ DP-Prompt is using a multilingual model (mt5-base) that supports German.")
    elif selected_anonymization == "DP-MLM":
        col3.success("✅ DP-MLM is using a German-specific BERT model for German text.")

### TEXT ANONYMISATION / PARAPHRASING ###

if send_button_clicked:

    if text_with_pii is None or len(text_with_pii.strip()) == 0: 
        st.error("Please upload a file or type in some text.")
    elif selected_anonymization is None: 
        st.error("Please select an anonymisation method.")
    else:
        col1, col2 = st.columns([1, 1])

        col1.subheader(f"📄 Original text")
        col1.text_area(
            label="Original Text:",
            value=text_with_pii,
            height=400,
            label_visibility="hidden" # visible, hidden, collapsed
        )
        # Store in session state
        st.session_state.text_with_pii = text_with_pii
        st.session_state.selected_anonymization = selected_anonymization

        if selected_anonymization in ["Diffractor", "DP-Prompt", "DP-MLM"] and selected_epsilon == 0:
            st.error("Please select a non-zero epsilon value.")
        else:
            # Add warning for German PII methods
            if selected_language == "German" and selected_anonymization in ["PII Deletion", "PII Labeling", "PII Replacement with Synthetic Data"]:
                with st.status("Preparing German language model..."):
                    st.write("Loading German NER model...")
                    try:
                        import spacy
                        if not spacy.util.is_package("de_core_news_lg"):
                            st.write("📥 German language model not found. Installing now...")
                            spacy.cli.download("de_core_news_lg")
                            st.write("✅ German model installed successfully!")
                        else:
                            st.write("✅ German model already installed!")
                    except Exception as e:
                        st.error(f"Error with German model: {str(e)}")
                        st.write("⚠️ Will fall back to English model for PII detection")
                
                st.warning("Note: German language support is experimental. If needed, the app will fall back to English-based detection while preserving German text content.")
            
            with st.spinner('Anonymizing text...'):
                try:
                    # Add progress placeholder for DP methods which can be slow
                    if selected_anonymization in ["DP-Prompt", "DP-MLM", "Diffractor"]:
                        progress_bar = st.progress(0)
                        progress_text = st.empty()
                        progress_text.text("Initializing models...")
                        
                        # For DP methods
                        if selected_anonymization == "Diffractor":
                            progress_text.text("Applying differential privacy with Diffractor...")
                            progress_bar.progress(30)
                            text_anonymized = diff_privacy_diffractor(text_with_pii, selected_epsilon, language=lang_code)
                            progress_bar.progress(100)
                            progress_text.text("Anonymization complete!")
                        elif selected_anonymization == "DP-Prompt":
                            progress_text.text("Applying DP-Prompt (this may take a while)...")
                            progress_bar.progress(30)
                            text_anonymized = diff_privacy_dp_prompt(text_with_pii, selected_epsilon, language=lang_code)
                            progress_bar.progress(100)
                            progress_text.text("Anonymization complete!")
                        elif selected_anonymization == "DP-MLM":
                            progress_text.text("Applying DP-MLM (this may take a while)...")
                            progress_bar.progress(30)
                            text_anonymized = diff_privacy_dpmlm(text_with_pii, selected_epsilon, language=lang_code)
                            progress_bar.progress(100)
                            progress_text.text("Anonymization complete!")
                    else:
                        # For non-DP methods
                        if selected_anonymization == "PII Deletion":
                            text_anonymized = delete_pii(text_with_pii, language=lang_code)
                        elif selected_anonymization == "PII Labeling":
                            text_anonymized = label_pii(text_with_pii, language=lang_code)
                        elif selected_anonymization == "PII Replacement with Synthetic Data":
                            text_anonymized = replace_pii(text_with_pii, language=lang_code)
                    
                    # Check if anonymization produced a result
                    if not text_anonymized or len(text_anonymized.strip()) == 0:
                        st.error("Anonymization process resulted in empty text. Please check your input text.")
                        text_anonymized = "Anonymization process did not produce a result."
                    
                    # Store anonymized text in session state
                    st.session_state.text_anonymized = text_anonymized
            
                    col2.subheader(f"🔐 Text after {selected_anonymization}")
                    col2.text_area(
                        label="Anonymized Text:",
                        value=text_anonymized,
                        height=400,
                        label_visibility="hidden" # visible, hidden, collapsed
                    )
                except Exception as e:
                    st.error(f"Error during anonymization: {str(e)}")
                    st_logger.error(f"Error during anonymization: {str(e)}", exc_info=True)
                    
                    # Display a more user-friendly error message based on the method
                    if selected_anonymization == "PII Replacement with Synthetic Data" and "OPENAI_API_KEY" in str(e):
                        st.warning("To use the PII Replacement with Synthetic Data method, you need to configure your OpenAI API key in the .env file.")
                    elif "CUDA" in str(e) or "GPU" in str(e):
                        st.warning("This method requires GPU resources that might not be available. Try using a different anonymization method.")
                    
                    # Reset anonymized text to prevent using incomplete results
                    st.session_state.text_anonymized = None

    if selected_anonymization in ["PII Deletion", "PII Labeling", "PII Replacement with Synthetic Data"]:
        try:
            st_analyze_results = analyze_text_with_presidio(text_with_pii, language=lang_code)
            results_as_dicts = [result.to_dict() for result in st_analyze_results]
            results_json = json.dumps(results_as_dicts, indent=2)

            with st.expander("Personally Identifiable Information (PII) detected in the original text", expanded=False):
                annotated_tokens = annotate(text=text_with_pii, analyze_results=st_analyze_results)
                annotated_text(*annotated_tokens)

            with st.expander("Detailed Findings", expanded=False):
                if results_json and st_analyze_results:
                    if len(st_analyze_results) > 0:
                        df_text = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
                        df_text["Text"] = [text_with_pii[res.start:res.end] for res in st_analyze_results]
                        df_subset = df_text[["entity_type", "Text", "start", "end", "score"]].rename(
                            {
                                "entity_type": "Entity type",
                                "start": "Start",
                                "end": "End",
                                "score": "Confidence",
                            },
                            axis=1,
                        )
                        st.dataframe(df_subset.reset_index(drop=True), use_container_width=True)
                    else:
                        st.text("No Personally identifiable information (PII) found")
                else:
                    st.text("No Personally identifiable information (PII) found")
        except Exception as e:
            st.error(f"Error displaying PII information: {str(e)}")
            st_logger.error(f"Error in PII display: {str(e)}", exc_info=True)
            st.warning("Continuing with anonymization, but PII detection results cannot be displayed.")
            
            # Log a warning but allow the app to continue
            st_logger.warning("PII detection issue but continuing with anonymization")

# Always display text boxes if they exist in session state (and not already shown by send_button)
if st.session_state.text_with_pii is not None and st.session_state.text_anonymized is not None and not send_button_clicked:
    col1, col2 = st.columns([1, 1])
    
    col1.subheader(f"📄 Original text")
    col1.text_area(
        label="Original Text:",
        value=st.session_state.text_with_pii,
        height=300,
        label_visibility="hidden",
        disabled=True
    )
    
    col2.subheader(f"🔐 Text after {st.session_state.selected_anonymization}")
    col2.text_area(
        label="Anonymized Text:",
        value=st.session_state.text_anonymized,
        height=300,
        label_visibility="hidden",
        disabled=True
    )

### RAG ###

st.markdown("---")
st.subheader("📝 RAG: ask a question about the text and compare the response based on the original text with the one based on the anonymized text")
st_logger.info("Waiting for user input...")
question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    st_logger.info("Get Answer button clicked.")

    # Check if text has been processed
    if st.session_state.text_with_pii is None:
        st.error("Please provide some text first and click 'Send' to anonymize it.")
    elif st.session_state.text_anonymized is None:
        st.error("Please anonymize your text by clicking 'Send' first.")
    elif not question:
        st.error("Please enter a question.")
    else:
        with st.spinner('Generating responses...'):
            response_original_obj = get_offline_RAG_response(question, st.session_state.text_with_pii)
            response_anonymized_obj = get_offline_RAG_response(question, st.session_state.text_anonymized)
            
            # Extract text content from response objects
            response_original = str(response_original_obj)
            response_anonymized = str(response_anonymized_obj)

        col1, col2 = st.columns([1, 1])
        col1.subheader(f"📄 Response based on the original text")
        col1.text_area(
            label="Response on the original text", 
            value=response_original, 
            height=200, 
            key="response_original",
            label_visibility="hidden" # visible, hidden, collapsed
        )

        col2.subheader(f"🔐 Response based on the text after {st.session_state.selected_anonymization}")
        col2.text_area(
            label="Response on the anonymized text", 
            value=response_anonymized, 
            height=200, 
            key="response_anonymized",
            label_visibility="hidden" # visible, hidden, collapsed
        )

### Evaluation ###

        rouge1_score = round(calculate_rouge1(response_original, response_anonymized), 2)
        rougeL_score = round(calculate_rougeL(response_original, response_anonymized), 2)
        bleu_score = round(calculate_bleu(response_original, response_anonymized), 2)
        cosine_similarity_score = round(calculate_cosine_similarity(response_original, response_anonymized), 2)
        perplexity_score = round(calculate_perplexity(response_anonymized), 2)
        llm_score = extract_llm_score(calculate_privacy_llm_judge(st.session_state.text_with_pii, response_anonymized))

        evaluation_df = pd.DataFrame({
            "Metric": [
                "Rouge-1", "Rouge-L", "BLEU", "Cosine Similarity", "Perplexity", "LLM-as-a-Judge"
            ],
            "Score": [
                rouge1_score, rougeL_score, bleu_score,
                cosine_similarity_score, perplexity_score, llm_score
            ],
            "Explanation": [
                "Overlap of unigrams (recall-focused)",
                "Longest common subsequence (sequence similarity)",
                "N-gram precision of generated vs reference",
                "Semantic closeness of embeddings",
                "How predictable the text is (lower = better)",
                "LLM-based judgment on percentage of privacy leakage"
            ]
        })

        # Normalize metrics where higher = better
        normalized_scores = []

        for metric, score in zip(evaluation_df["Metric"], evaluation_df["Score"]):
            if metric == "Perplexity":
                normalized = 1 / score if score > 0 else 0
            elif metric == "LLM-as-a-Judge":
                normalized = 1 - (score / 100)
            else:
                normalized = score  # assumed in 0-1 range
            normalized_scores.append(normalized)

        # Scale to [0, 1] range based on max
        evaluation_df["Normalized"] = normalized_scores
        evaluation_df["Normalized"] /= max(normalized_scores)

        col1.subheader(f"📊 Evaluation of the response based on the text after {st.session_state.selected_anonymization}")
        col1.dataframe(evaluation_df)

        bar_fig = px.bar(
            evaluation_df,
            x="Metric",
            y="Normalized",
            hover_data=["Explanation", "Score"],
            color="Metric",
            title="",
            labels={"Normalized": "Normalized Score (0–1)"},
        )
        col2.subheader(f"📊 Barchart for Normalized Evaluation Metrics")
        col2.plotly_chart(bar_fig, use_container_width=True)
