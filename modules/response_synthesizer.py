import os

from dotenv import load_dotenv
from gllm_generation.response_synthesizer import ResponseSynthesizer
from gllm_inference.request_processor import build_lm_request_processor
from gllm_generation.repacker import Repacker

load_dotenv()

SYSTEM_PROMPT = """
- Use only the information provided in the context below to answer the user's question.
You may infer simple, logical conclusions based on the context, but do not introduce
new facts or external knowledge.
- If the context does not contain enough information to answer the user's question, respond with:
"Sorry, I don't have enough information to answer that."

Context:
{context}
"""
USER_PROMPT = "Question: {query}"

lm_request_processor = build_lm_request_processor(
    model_id=os.environ["LANGUAGE_MODEL"],
    credentials=os.environ["OPENAI_API_KEY"],
    system_template=SYSTEM_PROMPT,
    user_template=USER_PROMPT,
)

response_synthesizer = ResponseSynthesizer.stuff(
    lm_request_processor=lm_request_processor,
    chunks_repacker=Repacker(mode="chunk"),
)
