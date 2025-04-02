import pandas as pd 
from RAG.Pinecone_LlamaIndex import getResponse

def generate_questions_basic(details, st_analyze_results):
    questions = []
    for entity in st_analyze_results:
        entity_type = entity.entity_type
        questions.append(f"What information related to {entity} is present in the text?")
        # questions.append(f"Can you identify any {entity_type} in the provided text?")
    return questions

def generate_questions_pii(raw_text, st_analyze_results):
    questions = []
    df = pd.DataFrame.from_records([r.to_dict() for r in st_analyze_results])
    df["Text"] = [raw_text[res.start:res.end] for res in st_analyze_results]
    
    # Loop through each row in the DataFrame and generate a question for each text
    for text in df["Text"]:
        questions.append(f"What information related to {text} is present in the text?")

    return questions

def evaluation(questions, file_hash):
    
    index_name = "masters-thesis-index"
    sum_results_with_pii = []
    sum_results_deleted = []
    sum_results_labeled = []
    sum_results_synthetic = []
    sum_results_dp = []

    for question in questions[1:5]:
        (response_with_pii, nodes_response_with_pii, evaluation_with_pii) = getResponse(index_name, question, [file_hash, "text_with_pii"])
        (response_without_pii, nodes_response_without_pii, evaluation_deleted) = getResponse(index_name, question, [file_hash, "text_pii_deleted"])
        (response_without_pii, nodes_response_without_pii, evaluation_labeled) = getResponse(index_name, question, [file_hash, "text_pii_labeled"])
        (response_without_pii, nodes_response_without_pii, evaluation_synthetic) = getResponse(index_name, question, [file_hash, "text_pii_synthetic"])
        (response_without_pii, nodes_response_without_pii, evaluation_dp) = getResponse(index_name, question, [file_hash, "text_pii_dp_prompt"])

        # Create a list of evaluator types
        evaluator_types = list(evaluation_with_pii.keys())

        # Create a list of results for each evaluation
        results_with_pii = [evaluation_with_pii[evaluator] for evaluator in evaluator_types]
        results_deleted = [evaluation_deleted[evaluator] for evaluator in evaluator_types]
        results_labeled = [evaluation_labeled[evaluator] for evaluator in evaluator_types]
        results_synthetic = [evaluation_synthetic[evaluator] for evaluator in evaluator_types]
        results_dp = [evaluation_dp[evaluator] for evaluator in evaluator_types]
        
        sum_results_with_pii = [sum_results_with_pii[0] + results_with_pii[0], sum_results_with_pii[1] + results_with_pii[1], sum_results_with_pii[2] + results_with_pii[2]]
        sum_results_deleted = [sum_results_deleted[0] + results_deleted[0], sum_results_deleted[1] + results_deleted[1], sum_results_deleted[2] + results_deleted[2]]
        sum_results_labeled = [sum_results_labeled[0] + results_labeled[0], sum_results_labeled[1] + results_labeled[1], sum_results_labeled[2] + results_labeled[2]]
        sum_results_synthetic = [sum_results_synthetic[0] + results_synthetic[0], sum_results_synthetic[1] + results_synthetic[1], sum_results_synthetic[2] + results_synthetic[2]]
        sum_results_dp = [sum_results_dp[0] + results_dp[0], sum_results_dp[1] + results_dp[1], sum_results_dp[2] + results_dp[2]]
    
    mean_results_with_pii = sum_results_with_pii / 5
    mean_results_deleted = sum_results_deleted / 5
    mean_results_labeled = sum_results_labeled / 5
    mean_results_synthetic = sum_results_synthetic / 5
    mean_results_dp = sum_results_dp / 5

    # Create a single DataFrame
    evaluation_df = pd.DataFrame({
        'Evaluator': evaluator_types,
        'Result with PII': mean_results_with_pii,
        'Result PII deleted': mean_results_deleted,
        'Result PII labeled': mean_results_labeled,
        'Result PII synthetic': mean_results_synthetic,
        'Result PII DP': mean_results_dp
    })

    return evaluation_df
