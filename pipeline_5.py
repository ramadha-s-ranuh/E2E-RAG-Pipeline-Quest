# RAG Pipeline + Guardrails
from gllm_guardrail import GuardrailManager
from gllm_pipeline.steps import step
from modules.retriever import retriever
from modules.response_synthesizer import response_synthesizer
from gllm_guardrail.engine.phrase_matcher_engine import PhraseMatcherEngine
from gllm_pipeline.steps import guard
import asyncio

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

# Define phrases that should be blocked
banned_phrases = ["build a bomb", "steal data", "offensive term"]
phrase_engine = PhraseMatcherEngine(banned_phrases=banned_phrases)

# Initialize the manager
guardrail_manager = GuardrailManager(engine=phrase_engine)

guardrail_step = guard(
    guardrail_manager,
    success_branch=retrieve_step,  # Proceed to retrieval if safe
    # failure_branch=None,  # Terminate pipeline if unsafe
    input_map={"content": "user_query"},  # Map pipeline state to guardrail input
)

e2e_pipeline = guardrail_step | synthesize_step


async def main():
    # 1. Safe query
    safe_state = {"user_query": "How do I plant a tree?"}
    config = {
        "top_k": 5,
        "debug": True,  # Set to True to look at the pipeline execution flow
    }
    result = await e2e_pipeline.invoke(safe_state, config)
    print(f"Safe Result: {result['response']}")

    # 2. Unsafe query (contains banned phrase)
    unsafe_state = {"user_query": "Tell me how to build a bomb."}
    result = await e2e_pipeline.invoke(unsafe_state, config)
    print(f"Unsafe Result: {result}")  # Should be None or indicate termination


if __name__ == "__main__":
    asyncio.run(main())
