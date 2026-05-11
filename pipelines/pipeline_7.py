# RAG Pipeline + Multimodal Handling
import asyncio
import os
from typing import Any
from gllm_inference.schema import MessageContent, Attachment
from gllm_pipeline.pipeline import RAGState
from gllm_pipeline.steps import step, transform
from gllm_generation.response_synthesizer import (
    ResponseSynthesizer,
)
from gllm_inference.request_processor import build_lm_request_processor
from modules.retriever import retriever
from dotenv import load_dotenv

load_dotenv()


class MultimodalRAGState(RAGState):
    attachments: list[str]
    extra_contents: list[MessageContent]


def format_extra_contents(inputs: dict[str, Any]) -> list[MessageContent]:
    attachments: list[bytes] = inputs["attachments"]
    return [Attachment.from_path(path) for path in attachments]


response_synthesizer = ResponseSynthesizer.stuff(
    lm_request_processor=build_lm_request_processor(
        model_id=os.getenv("LANGUAGE_MODEL"),
        credentials=os.getenv("OPENAI_API_KEY"),
        system_template="""Find the imaginary animal that is similar to the animal in the picture. Context: {context}""",
        user_template="Question: {query}",
    )
)

format_extra_contents_step = transform(  # 👈 New step
    format_extra_contents,
    ["attachments"],
    "extra_contents",
)

response_synthesizer_step = step(
    response_synthesizer,
    {
        "query": "user_query",
        "chunks": "chunks",
        "extra_contents": "extra_contents",  # 👈 New parameter
    },
    "response",
)

retrieve_step = step(
    component=retriever,
    input_map={"query": "user_query", "top_k": "top_k"},
    output_state="chunks",
)


e2e_pipeline = format_extra_contents_step | retrieve_step | response_synthesizer_step
e2e_pipeline.state_type = MultimodalRAGState


async def main():
    state = {
        "user_query": "Find the animal in the chunks that has the most similar description to the image.",
        "attachments": ["dog2.png"],
    }
    config = {"top_k": 5}
    result = await e2e_pipeline.invoke(state, config)
    print(f"Pipeline result: {result['response']}")


if __name__ == "__main__":
    asyncio.run(main())
