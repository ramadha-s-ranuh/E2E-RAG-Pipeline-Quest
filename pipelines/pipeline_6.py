# Regular RAG Pipeline + Query Transformation
import asyncio
from gllm_pipeline.steps import step, transform
from gllm_pipeline.pipeline import RAGState
from gllm_retrieval.query_transformer.one_to_one_query_transformer import (
    OneToOneQueryTransformer,
)
from gllm_inference.request_processor import build_lm_request_processor
from modules.retriever import retriever
from modules.response_synthesizer import response_synthesizer


class RAGStateWithQT(RAGState):
    query: str


transform_query_step = step(
    component=OneToOneQueryTransformer(
        lm_request_processor=build_lm_request_processor(
            model_id="openai/gpt-4o-mini",
            system_template="You are a helpful assistant that rewrites queries for better retrieval. Rewrite the following query. Only output the transformed query.",
            user_template="Query: {query}",
        )
    ),
    input_map={"query": "user_query"},
    output_state="queries",
)

flatten_query = transform(
    operation=lambda x: "\n".join(x["queries"]),
    input_states=["queries"],
    output_state="query",
)

retrieve_step = step(
    component=retriever,
    input_map={"query": "user_query", "top_k": "top_k"},
    output_state="chunks",
)

response_synthesizer_step = step(
    component=response_synthesizer,
    input_map={"query": "user_query", "chunks": "chunks"},
    output_state="response",
)

state = {
    "user_query": "Give me nocturnal creatures from the dataset",  # Replace with your actual query
}

config = {
    "top_k": 5,
    "debug": True,  # Set to True to look at the pipeline execution flow
}

e2e_pipeline = (
    transform_query_step | flatten_query | retrieve_step | response_synthesizer_step
)

e2e_pipeline.state_type = RAGStateWithQT

result = asyncio.run(e2e_pipeline.invoke(state, config))
print(f"Pipeline result: {result['response']}")
