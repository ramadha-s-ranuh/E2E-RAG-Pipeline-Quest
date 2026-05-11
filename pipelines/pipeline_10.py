from typing import TypedDict
import asyncio
from gllm_pipeline.pipeline import Pipeline
from gllm_pipeline.steps import step, subgraph


# Example dummy components - replace with your actual implementations
class QueryProcessor:
    pass


class QueryExpander:
    pass


class DocumentRetriever:
    pass


class DocumentFilter:
    pass


class RelevanceReranker:
    pass


class TopKSelector:
    pass


class ContextBuilder:
    pass


class PromptBuilder:
    pass


class LLMGenerator:
    pass


class ResponseFormatter:
    pass


class ResponseValidator:
    pass


class MetadataExtractor:
    pass


# Clean main pipeline state - only essential variables
class MainRAGState(TypedDict):
    user_query: str
    processed_query: str
    expanded_query: str
    context: str
    final_response: str
    metadata: dict


class ModularRAGPipelineBuilder:
    """Modular RAG pipeline builder using subgraphs."""

    def build(self) -> Pipeline:
        """Build the main pipeline using subgraphs."""
        preprocessing_step = self._build_preprocessing_subgraph()
        retrieval_step = self._build_retrieval_subgraph()
        generation_step = self._build_generation_subgraph()

        pipeline = Pipeline(
            steps=[
                preprocessing_step,
                retrieval_step,
                generation_step,
            ],
            state_type=MainRAGState,
            recursion_limit=100,
        )

        return pipeline


def _build_retrieval_subgraph(self):
    """Build the document retrieval subgraph."""

    class RetrievalState(TypedDict):
        query: str
        retrieved_documents: list
        filtered_documents: list
        reranked_documents: list
        selected_documents: list
        context: str

    # Create the retrieval pipeline
    retrieval_pipeline = Pipeline(
        [
            step(DocumentRetriever(), {"query": "query"}, "retrieved_documents"),
            step(
                DocumentFilter(),
                {"documents": "retrieved_documents"},
                "filtered_documents",
            ),
            step(
                RelevanceReranker(),
                {"documents": "filtered_documents", "query": "query"},
                "reranked_documents",
            ),
            step(
                TopKSelector(),
                {"documents": "reranked_documents"},
                "selected_documents",
            ),
            step(ContextBuilder(), {"documents": "selected_documents"}, "context"),
        ],
        state_type=RetrievalState,
    )

    return subgraph(
        subgraph=retrieval_pipeline,
        input_map={"query": "expanded_query"},
        output_state_map={"context": "context"},
        name="retrieval_step",
    )


def _build_generation_subgraph(self):
    """Build the response generation subgraph."""

    class GenerationState(TypedDict):
        query: str
        context: str
        prompt: str
        generated_response: str
        formatted_response: str
        validated_response: str
        response_metadata: dict

    # Create the generation pipeline
    generation_pipeline = Pipeline(
        [
            step(PromptBuilder(), {"query": "query", "context": "context"}, "prompt"),
            step(LLMGenerator(), {"prompt": "prompt"}, "generated_response"),
            step(
                ResponseFormatter(),
                {"response": "generated_response"},
                "formatted_response",
            ),
            step(
                ResponseValidator(),
                {"response": "formatted_response"},
                "validated_response",
            ),
            step(
                MetadataExtractor(),
                {"response": "validated_response"},
                "response_metadata",
            ),
        ],
        state_type=GenerationState,
    )

    return subgraph(
        subgraph=generation_pipeline,
        input_map={"query": "processed_query", "context": "context"},
        output_state_map={
            "final_response": "validated_response",
            "metadata": "response_metadata",
        },
        name="generation_step",
    )


# Test the pipeline
async def test_modular_pipeline():
    builder = ModularRAGPipelineBuilder()
    pipeline = builder.build()

    state = {
        "user_query": "What are some forest animals?",
    }

    config = {
        "top_k": 5,
        "debug": True,
    }

    result = await pipeline.invoke(state, config)
    print(f"Pipeline result: {result}")


if __name__ == "__main__":
    asyncio.run(test_modular_pipeline())
