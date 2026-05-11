# Regular RAG Pipeline
import asyncio
from gllm_pipeline.steps import step
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

state = {
    "user_query": "Give me nocturnal creatures from the dataset",
    "event_emitter": None,
}

config = {
    "top_k": 5,
    "debug": True,
}

e2e_pipeline = retrieve_step | response_synthesizer_step

result = asyncio.run(e2e_pipeline.invoke(state, config))
print(f"Pipeline result: {result['response']}")
