# RAG Pipeline + Document References
import os
import asyncio
from gllm_generation.reference_formatter import SimilarityBasedReferenceFormatter
from gllm_inference.em_invoker import build_em_invoker
from gllm_pipeline.steps import step
from modules.retriever import retriever
from modules.response_synthesizer import response_synthesizer

em_invoker = build_em_invoker(
    "openai/text-embedding-3-small", credentials=os.getenv("OPENAI_API_KEY")
)

retrieve_step = step(
    component=retriever,
    input_map={"query": "user_query", "top_k": "top_k"},
    output_state="chunks",
)

synthesize_step = step(
    component=response_synthesizer,
    input_map={"query": "user_query", "chunks": "chunks"},
    output_state="response",
)

state = {
    "user_query": "Give me nocturnal creatures from the dataset",  # Replace with your actual query
    "event_emitter": None,
}

config = {
    "top_k": 5,
    "debug": True,  # Set to True to look at the pipeline execution flow
}

reference_formatter = SimilarityBasedReferenceFormatter(
    em_invoker=em_invoker, threshold=0.5, stringify=False
)

format_reference_step = step(
    component=reference_formatter,
    input_map={"response": "response", "chunks": "chunks"},
    output_state="references",
)

e2e_pipeline = retrieve_step | synthesize_step | format_reference_step


async def main():
    state = {
        "user_query": "Give me nocturnal creatures from the dataset"
    }  # Replace with your actual query
    config = {"top_k": 5}
    result = await e2e_pipeline.invoke(state, config)
    print(f"Pipeline result: {result['response']}")
    print(f"References: {result['references']}")


if __name__ == "__main__":
    asyncio.run(main())
