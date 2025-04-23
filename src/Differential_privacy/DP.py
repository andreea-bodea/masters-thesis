from .DPMLM.DPMLM import DPMLM
from .Diffractor.Diffractor import Lists, Diffractor
from .PrivFill.LLMDP import DPPrompt
import nltk
from nltk.data import find
import os

def diff_privacy_diffractor(text_with_pii, epsilon, language="en"):
    try:
        find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Get the absolute path to the data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Diffractor/data')
    
    lists = Lists(
#       home="./src/Differential_privacy/Diffractor/data",
        home=data_dir,
    )
    diff = Diffractor(
        L=lists,
        gamma=5,
        epsilon=epsilon,
        rep_stop=False, 
        method="geometric"
    )
    text_lower_case = text_with_pii.lower()
    perturbed_text, num_perturbed, num_diff, total, support, changes = diff.rewrite(text_lower_case)
    diff.cleanup()
    return ' '.join(perturbed_text)

def diff_privacy_dp_prompt(text_with_pii, epsilon, language="en"):
    sentences = nltk.sent_tokenize(text_with_pii, language=language)
    # For German language, use a multilingual model
    if language == "de":
        model_checkpoint = "google/mt5-base"
    else:
        model_checkpoint = "google/flan-t5-large"
    
    dpprompt = DPPrompt(model_checkpoint=model_checkpoint)
    text_pii_dp_dp_prompt = dpprompt.privatize_dp(sentences, epsilon=epsilon)
    return ' '.join(text_pii_dp_dp_prompt)

def diff_privacy_dpmlm(text_with_pii, epsilon, language="en"):
    sentences = nltk.sent_tokenize(text_with_pii, language=language)
    # Initialize DPMLM with appropriate language model
    if language == "de":
        dpmlm = DPMLM(model_name="dbmdz/bert-base-german-cased")
    else:
        dpmlm = DPMLM()
    
    perturbed_sentences = []
    
    for sentence in sentences:
        # If sentence is too long, break it into chunks
        tokens = nltk.word_tokenize(sentence, language=language)
        if len(tokens) > 75:  # Conservative estimate for model's token limit
            # Process in fixed-size chunks
            chunk_size = 75
            chunks = []
            
            # Split into chunks
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i:i + chunk_size]
                chunks.append(' '.join(chunk))
            
            # Process each chunk
            chunk_results = []
            for chunk in chunks:
                perturbed_chunk, perturbed, total = dpmlm.dpmlm_rewrite(chunk, epsilon=epsilon)
                chunk_results.append(perturbed_chunk)
            
            # Join chunk results and append to perturbed_sentences
            perturbed_sentences.append(' '.join(chunk_results))
        else:
            # Process normally if sentence is not too long
            perturbed_sentence, perturbed, total = dpmlm.dpmlm_rewrite(sentence, epsilon=epsilon)
            perturbed_sentences.append(perturbed_sentence)
    
    text_pii_dpmlm = ' '.join(perturbed_sentences)
    return text_pii_dpmlm

if __name__ == "__main__":

    # text = "This is a sample text containing sensitive information like a phone number 123-456-7890."
    # text = "David"
    # text = "Arthur Hailey: King of the bestsellers Novelist Arthur Hailey, who has died at the age of 84, was known for his bestselling page-turners exploring the inner workings of various industries, from the hotels to high finance. Born in Luton, Bedfordshire, on 5 April 1920, Hailey was the only child of working class parents, They could not afford to keep him in school beyond the age of 14. He served as a pilot with the Royal Air Force during World War II, flying fighter planes to the Middle East. It was an occupation that was later to feature in his authorial debut, the television screenplay Flight into Danger. Hailey emigrated to Canada in 1947, where he eventually became a citizen. He wanted to be a writer from an early age, but did not take it up professionally until his mid-thirties, when he was inspired to write his first screenplay while on a return flight to Toronto."
    text = "Ireland 19-13 England Ireland consigned England to their third straight Six Nations defeat with a stirring victory at Lansdowne Road. A second-half try from captain Brian O'Driscoll and 14 points from Ronan O'Gara kept Ireland on track for their first Grand Slam since 1948. England scored first through Martin Corry but had tries from Mark Cueto and Josh Lewsey disallowed. Andy Robinson's men have now lost nine of their last 14 matches since the 2003 World Cup final. The defeat also heralded England's worst run in the championship since 1987. Ireland last won the title, then the Five Nations, in 1985, but 20 years on they share top spot in the table on maximum points with Wales."
    
    """
    # Diffractor
    print("Initial text:", text)
    epsilon = 1.5
    result_diffractor = diff_privacy_diffractor(text, epsilon) 
    print("Epsilon:", epsilon)
    print("Perturbed Text:", result_diffractor)
    """ 
    """   
    # DP Prompt
    print("Initial text:", text)
    epsilon = 1000
    result_dp_prompt = diff_privacy_dp_prompt(text, epsilon)
    print("Epsilon:", epsilon)
    print("Perturbed Text:")
    sentences = nltk.sent_tokenize(result_dp_prompt)
    for sentence in sentences:
        print(sentence)
    """
    
    # DP MLM
    print("Initial text:", text)
    epsilon = 100
    result_dpmlm = diff_privacy_dpmlm(text, epsilon)
    print("Epsilon:", epsilon)
    print("Perturbed Text:", result_dpmlm)