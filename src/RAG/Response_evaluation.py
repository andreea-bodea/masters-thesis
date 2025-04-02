import logging
from Data.Database_management import retrieve_responses_by_name_and_question, update_response_evaluation
from Presidio.Presidio_helpers import analyze, analyzer_engine
from Response_generation import get_all_questions
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import math
import pandas as pd

st_logger = logging.getLogger('Response evaluation')
st_logger.setLevel(logging.INFO)

anonymization_types = ['response_pii_deleted', 'response_pii_labeled', 'response_pii_synthetic', 'response_diffractor', 'response_dp_prompt', 'response_dpmlm']
questions = get_all_questions()
question_utility = questions[0]
question_untargeted_attack = questions[1]

def calculate_rouge1(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    rouge1_fmeasure = scores['rouge1'].fmeasure
    return rouge1_fmeasure

def calculate_rougeL(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) # ['rouge1', 'rouge2', 'rougeL']
    scores = scorer.score(reference, hypothesis)
    rougeL_fmeasure = scores['rougeL'].fmeasure
    return rougeL_fmeasure

def calculate_bleu(reference, hypothesis):
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing_function)
    return bleu_score

def calculate_cosine_similarity(reference, hypothesis):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    ref_embedding = model.encode([reference])
    hyp_embedding = model.encode([hypothesis])
    cosine_sim = cosine_similarity(ref_embedding, hyp_embedding)
    return cosine_sim[0][0]

def calculate_perplexity(text):
    # Load the GPT-2 model and tokenizer
    model_name = 'gpt2'
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    encodings = tokenizer(text, return_tensors='pt', truncation=True, max_length=model.config.n_positions)
    max_length = model.config.n_positions
    stride = 512  # adjust based on GPU/memory constraints
    nlls = []
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)

    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i  # number of tokens to predict
        input_ids_slice = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_slice.clone()
        target_ids[:, :-trg_len] = -100  # only compute loss on the target tokens

        with torch.no_grad():
            outputs = model(input_ids_slice, labels=target_ids)
            # Multiply loss by target length to get total loss for this slice
            nll = outputs.loss * trg_len
            nlls.append(nll)

    # Sum losses and divide by total tokens to get average loss per token
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / seq_len)
    return ppl.item()

def calculate_reduction_in_pii():
    for response_type in response_types:
        st_logger.info(f"Presidio text analysis started on the text: {response_type}")
        analyzer = analyzer_engine()
        st_analyze_results = analyze(
            text=database_file[response_type],
            language="en",
            score_threshold=0.5,
            allow_list=[],
    )
    #st_logger.info(f"Presidio text analysis completed {st_analyze_results}")
    results_as_dicts = [result.to_dict() for result in st_analyze_results]
    #st_logger.info(f"Presidio text analysis completed {st_analyze_results}")
    pii_in_response[response_type].append(len(results_as_dicts))
    print(pii_in_response)
    # REDUCTION IN PII COUNT
    reduction_in_pii = {}
    for response_type in response_types:
        if response_type != 'response_with_pii':
            reduction_in_pii[response_type] = [
                original - anonymized for original, anonymized in zip(pii_in_response['response_with_pii'], pii_in_response[response_type])
            ]
            print(f"Reduction in PII for {response_type}: {reduction_in_pii[response_type]}")

def evaluate(table_name, file_name, type):
    database_file = retrieve_responses_by_name_and_question(table_name, file_name, question_utility)  
    if database_file is None:
        st_logger.error(f"No data found for file: {file_name} and provided question.")
    else: 
        scores = {}
        if type == "utility":
            for anonymization_type in anonymization_types:
                rouge_score1 = calculate_rouge1(database_file['response_with_pii'], database_file[anonymization_type])
                rouge_scoreL = calculate_rougeL(database_file['response_with_pii'], database_file[anonymization_type])
                bleu_score = calculate_bleu(database_file['response_with_pii'], database_file[anonymization_type])
                cosine_sim = calculate_cosine_similarity(database_file['response_with_pii'], database_file[anonymization_type])
                perplexity = calculate_perplexity(database_file[anonymization_type])
                scores[anonymization_type] = {
                    'rouge_score1': rouge_score1,
                    'rouge_scoreL': rouge_scoreL,
                    'bleu_score': bleu_score,
                    'cosine_similarity': cosine_sim,
                    'perplexity': perplexity
                }
            st_logger.error(f"Utility evaluation scores for {file_name}: {scores}")
            update_response_evaluation(table_name=table_name, file_name=file_name, question=question_utility, evaluation=scores)
        elif type == "privacy":
            for anonymization_type in anonymization_types:
                # Placeholder for privacy evaluation logic
                pass
            st_logger.error(f"Privacy evaluation scores for {file_name}: {scores}")
            update_response_evaluation(table_name=table_name, file_name=file_name, question=question_untargeted_attack, evaluation=scores)

def evaluate_all(table_name, file_name_pattern, type, last): 
    for i in range(1, last+1):  # FOR EACH DATABASE FILE
        file_name = file_name_pattern.format(i)
        if type == "utility":
            evaluate(table_name, file_name, type="utility")
        elif type == "privacy":
            evaluate(table_name, file_name, type="privacy")

def average_utility(table_name, file_name_pattern, last):
    rouge1_scores = {key: [] for key in anonymization_types}
    rougeL_scores = {key: [] for key in anonymization_types}
    bleu_scores = {key: [] for key in anonymization_types}
    cosine_sim_scores = {key: [] for key in anonymization_types}
    perplexity_scores = {key: [] for key in anonymization_types}

    for i in range(1, last+1):  # FOR EACH DATABASE FILE
        file_name = file_name_pattern.format(i)
        database_file = retrieve_responses_by_name_and_question(table_name, file_name, question_utility)  
        if database_file and 'evaluation' in database_file:
            evaluation = database_file['evaluation']
            for anonymization_type in anonymization_types:
                if anonymization_type in evaluation:
                    rouge1_scores[anonymization_type].append(evaluation[anonymization_type]['rouge_score1'])
                    rougeL_scores[anonymization_type].append(evaluation[anonymization_type]['rouge_scoreL'])
                    bleu_scores[anonymization_type].append(evaluation[anonymization_type]['bleu_score'])
                    cosine_sim_scores[anonymization_type].append(evaluation[anonymization_type]['cosine_similarity'])
                    perplexity_scores[anonymization_type].append(evaluation[anonymization_type]['perplexity'])

    final_rouge1_score = {key: sum(scores) / len(scores) for key, scores in rouge1_scores.items() if scores}
    final_rougeL_score = {key: sum(scores) / len(scores) for key, scores in rougeL_scores.items() if scores}
    final_bleu_score = {key: sum(scores) / len(scores) for key, scores in bleu_scores.items() if scores}
    final_cosine_sim_score = {key: sum(scores) / len(scores) for key, scores in cosine_sim_scores.items() if scores}
    final_perplexity_score = {key: sum(scores) / len(scores) for key, scores in perplexity_scores.items() if scores}

    for anonymization_type in anonymization_types:
        print(f"Final average ROUGE-1 score for {anonymization_type}: {final_rouge1_score.get(anonymization_type, 'No scores available')}")
        print(f"Final average ROUGE-L score for {anonymization_type}: {final_rougeL_score.get(anonymization_type, 'No scores available')}")
        print(f"Final average BLEU score for {anonymization_type}: {final_bleu_score.get(anonymization_type, 'No scores available')}")
        print(f"Final average Cosine Similarity score for {anonymization_type}: {final_cosine_sim_score.get(anonymization_type, 'No scores available')}")
        print(f"Final average Perplexity score for {anonymization_type}: {final_perplexity_score.get(anonymization_type, 'No scores available')}")

    scores_df = pd.DataFrame({
        'Anonymization Type': anonymization_types,
        'Average ROUGE-1 Score': [final_rouge1_score.get(at, 'No scores available') for at in anonymization_types],
        'Average ROUGE-L Score': [final_rougeL_score.get(at, 'No scores available') for at in anonymization_types],
        'Average BLEU Score': [final_bleu_score.get(at, 'No scores available') for at in anonymization_types],
        'Average Cosine Similarity Score': [final_cosine_sim_score.get(at, 'No scores available') for at in anonymization_types],
        'Average Perplexity Score': [final_perplexity_score.get(at, 'No scores available') for at in anonymization_types]
    })
    print(scores_df)
    return scores_df

def average_privacy(table_name, file_name_pattern, last):
    untargeted_attack_scores = {key: [] for key in anonymization_types}

    for i in range(1, last+1):  # FOR EACH DATABASE FILE
        file_name = file_name_pattern.format(i)
        database_file = retrieve_responses_by_name_and_question(table_name, file_name, question_untargeted_attack)  
        if database_file and 'evaluation' in database_file:
            evaluation = database_file['evaluation']
            for anonymization_type in anonymization_types:
                if anonymization_type in evaluation:
                    untargeted_attack_scores[anonymization_type].append(evaluation[anonymization_type]['untargeted_attack_scores'])

    final_reduction_in_pii_score = {response_type: sum(reductions) / len(reductions) if reductions else 0
                         for response_type, reductions in reduction_in_pii.items()}
    
    for anonymization_type in anonymization_types:
        print(f"Final average reduction in pii for {anonymization_type}: {final_reduction_in_pii_score.get(anonymization_type, 'No scores available')}")

    scores_df = pd.DataFrame({
        'Anonymization Type': anonymization_types,
        'Average Reduction in PII': [final_reduction_in_pii_score.get(at, 'No scores available') for at in anonymization_types],
        })
    print(scores_df)
    return scores_df

if __name__ == "__main__":
    evaluate_all(table_name="enron_responses2", file_name_pattern="Enron_{}", type="utility", last=100)
    evaluate_all(table_name="enron_responses2", file_name_pattern="Enron_{}", type="privacy", last=100)

    evaluate_all(table_name="bbc_responses2", file_name_pattern="BBC_{}", type="utility", last=200)
    evaluate_all(table_name="bbc_responses2", file_name_pattern="BBC_{}", type="privacy", last=200)

    average_utility(table_name="enron_responses2", file_name_pattern="Enron_{}", first=1, last=60)
    average_utility(table_name="bbc_responses2", file_name_pattern="BBC_{}", first=1, last=200)