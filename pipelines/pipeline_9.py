# RAG with Dynamic Models
import asyncio
from gllm_pipeline.steps import step
from gllm_generation.response_synthesizer import ResponseSynthesizer
from gllm_pipeline.pipeline import Pipeline
from modules.retriever import retriever
from modules.response_synthesizer import response_synthesizer

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


def build_response_synthesizer(model_id: str) -> ResponseSynthesizer:
    """Build a response synthesizer for the given model.

    Args:
        model_id (str): The model identifier to use for the LM request processor.

    Returns:
        ResponseSynthesizer: Synthesizer configured with the given model.
    """
    return ResponseSynthesizer.preset.stuff(model_id)


def build_pipeline(model_id: str) -> Pipeline:
    """Build the end-to-end pipeline.

    Args:
        model_id (str): Model identifier used to build the response synthesizer.

    Returns:
        Any: A composed pipeline with .invoke(state, config) coroutine method.
    """
    # The following steps stay the same
    retriever_step = step(
        retriever,
        input_map={"query": "user_query", "top_k": "top_k"},
        output_state="chunks",
    )

    response_synthesizer_step = step(
        component=response_synthesizer,
        input_map={
            "query": "user_query",
            "chunks": "chunks",
        },
        output_state="response",
    )
    return retriever_step | response_synthesizer_step


if __name__ == "__main__":
    model_id = "openai/gpt-5-nano"  # Change this to whatever you want
    e2e_pipeline = build_pipeline(model_id)
    state = {
        "user_query": "Give me nocturnal creature from the dataset",  # Replace with your actual query
    }

    config = {
        "top_k": 5,
        "debug": True,  # Set to True to look at the pipeline execution flow
    }

    result = asyncio.run(e2e_pipeline.invoke(state, config))
    print(f"Response: {result['response']}")
