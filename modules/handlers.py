import os

from gllm_inference.request_processor import build_lm_request_processor
from gllm_generation.response_synthesizer import ResponseSynthesizer

response_synthesizer_general = ResponseSynthesizer.stuff(
    lm_request_processor=build_lm_request_processor(
        model_id="openai/gpt-5-nano",
        credentials=os.getenv("OPENAI_API_KEY"),
        system_template="You are a helpful assistant that answers general knowledge questions.",
        user_template="{query}",
    )
)
