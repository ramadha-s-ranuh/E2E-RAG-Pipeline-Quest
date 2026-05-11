# RAG Pipeline + Dynamic Step
import asyncio
from gllm_pipeline.steps import step, toggle
from gllm_pipeline.pipeline.states import RAGState
from modules.retriever import retriever
from modules.response_synthesizer import response_synthesizer

retriever_step = step(
    component=retriever,
    input_map={"query": "user_query", "top_k": "top_k"},
    output_state="chunks",
)

response_synthesizer_step = step(
    component=response_synthesizer,
    input_map={"query": "user_query", "chunks": "chunks"},
    output_state="response",
)


class ToogleState(RAGState):
    use_knowledge_base: bool


knowledge_base_toggle_step = toggle(
    condition=lambda x: x["use_knowledge_base"],
    if_branch=[retriever_step],
)

e2e_pipeline = retriever_step | response_synthesizer_step
e2e_pipeline.state_tpye = ToogleState


async def main():
    state = {
        "user_query": "Give me all nocturnal creatures from the dataset",
        "use_knowledge_base": False,
        "chunks": [],
    }

    config = {
        "top_k": 5,
        "debug": True,
    }

    result = await e2e_pipeline.invoke(state, config)
    print(f"Pipeline result: {result['response']}")


if __name__ == "__main__":
    asyncio.run(main())
