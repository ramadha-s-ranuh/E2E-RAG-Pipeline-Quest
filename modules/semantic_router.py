import os
import json

from dotenv import load_dotenv

# from gllm_inference.em_invoker import build_em_invoker
from gllm_pipeline.router import AurelioSemanticRouter
from semantic_router.encoders import OpenAIEncoder

load_dotenv()

# Create embedding model invoker
encoder = OpenAIEncoder(
    openai_api_key=os.getenv("OPENAI_API_KEY"), name="text-embedding-3-small"
)

# Load route examples from JSON file
with open("route_examples.json", "r", encoding="utf-8") as f:
    route_examples_data = json.load(f)

# Convert JSON format to route_examples dict
route_examples = {route["name"]: route["utterances"] for route in route_examples_data}

# Create semantic router with Aurelio backend
semantic_router = AurelioSemanticRouter(
    default_route="general",
    valid_routes={"knowledge_base", "general"},
    encoder=encoder,
    routes=route_examples,
)
