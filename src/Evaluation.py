def evaluateRAG(index_name: str, text: str) -> 
    
    document = Document(text=text)
    
    data_generator = DatasetGenerator.from_documents(documents=[document])
    eval_questions = data_generator.generate_questions_from_nodes()
    gpt4 = OpenAI(temperature=0, model="gpt-4")
    evaluator_gpt4 = RelevancyEvaluator(llm=gpt4)
    
    # create vector index
    vector_index = VectorStoreIndex.from_documents(document)

    query_engine = vector_index.as_query_engine()
    response_vector = query_engine.query(eval_questions[1])
    eval_result = evaluator_gpt4.evaluate_response(
        query=eval_questions[1], response=response_vector
    )
    return eval_result 



data_generator = DatasetGenerator.from_documents(documents)

eval_questions = data_generator.generate_questions_from_nodes()

eval_questions

# gpt-4
gpt4 = OpenAI(temperature=0, model="gpt-4")

evaluator_gpt4 = RelevancyEvaluator(llm=gpt4)

# create vector index
vector_index = VectorStoreIndex.from_documents(documents)

# define jupyter display function
def display_eval_df(query: str, response: Response, eval_result: str) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Response": str(response),
            "Source": (
                response.source_nodes[0].node.get_content()[:1000] + "..."
            ),
            "Evaluation Result": eval_result,
        },
        index=[0],
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"]
    )
    display(eval_df)

query_engine = vector_index.as_query_engine()
response_vector = query_engine.query(eval_questions[1])
eval_result = evaluator_gpt4.evaluate_response(
    query=eval_questions[1], response=response_vector
)

display_eval_df(eval_questions[1], response_vector, eval_result)

