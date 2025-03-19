from Diffractor.Diffractor import Lists, Diffractor
from LLMDP import DPPrompt
import nltk
from nltk.data import find

def diff_privacy_dp_prompt(text_with_pii):
        # def split_text_into_chunks(text, max_tokens=512):
        # Tokenize the text to count tokens
        # tokens = dpprompt.tokenizer.tokenize(text)
        # for i in range(0, len(tokens), max_tokens):
        #    yield dpprompt.tokenizer.convert_tokens_to_string(tokens[i:i + max_tokens])

    # Split the long text into manageable chunks
    # text_chunks = list(split_text_into_chunks(text_with_pii, max_tokens=512))
    
    dpprompt = DPPrompt(model_checkpoint="google/flan-t5-large")
    text_pii_dp_prompt = dpprompt.privatize(text_with_pii, epsilon=200)
    return text_pii_dp_prompt

def diff_privacy_diffractor(text_with_pii):
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    lists = Lists(
        home="/Users/andreeabodea/ANDREEA/MT/Code/masters-thesis/Diffractor",
    )
    diff = Diffractor(
        L=lists,
        gamma=5,
        epsilon=1,
        rep_stop=False, 
        method="geometric"
    )
    text_lower_case = text_with_pii.lower()
    perturbed_text, num_perturbed, num_diff, total, support, changes = diff.rewrite(text_lower_case)
    diff.cleanup()
    return perturbed_text 