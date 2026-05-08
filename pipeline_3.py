# RAG Pipeline + Semantic Routing
import asyncio
from gllm_pipeline.steps import step
from modules.retriever import retriever
from modules.response_synthesizer import response_synthesizer
from modules.handlers import response_synthesizer_general
from modules.semantic_router import semantic_router
from gllm_pipeline.steps import switch
from gllm_pipeline.pipeline import RAGState, Pipeline

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

synthesize_general_step = step(
    component=response_synthesizer_general,
    input_map={
        "query": "user_query",
    },
    output_state="response",
)


# Implement switch step to wrap the pipeline with semantic router as condition
conditional_step = switch(
    condition=semantic_router,
    branches={
        "knowledge_base": [retrieve_step, synthesize_step],
        "general": synthesize_general_step,
    },
    default=synthesize_general_step,
    input_map={"source": "user_query"},
    output_state="response",
)

# Define state type for the pipeline, extend RAGState


class RouterState(RAGState):
    route: str
    source: str


# Initialize the pipeline with the conditional step and the state type
e2e_pipeline = Pipeline(steps=[conditional_step], state_type=RouterState)


async def main():
    state = {
        "user_query": "What is the capital of Japan?"
    }  # Replace with your actual query
    config = {"top_k": 5}
    result = await e2e_pipeline.invoke(state, config)
    print(f"Pipeline result: {result['response']}")


if __name__ == "__main__":
    asyncio.run(main())
