import os

from dotenv import load_dotenv
from gllm_datastore.data_store import ChromaDataStore
from gllm_datastore.data_store.chroma.data_store import ChromaClientType
from gllm_inference.em_invoker.openai_em_invoker import OpenAIEMInvoker
from gllm_retrieval.retriever import VectorRetriever

load_dotenv()

embedding_model = OpenAIEMInvoker(
    model_name=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

data_store = ChromaDataStore(
    collection_name="documents",
    client_type=ChromaClientType.PERSISTENT,
    persist_directory="data",
).with_vector(em_invoker=embedding_model)


retriever = VectorRetriever(data_store=data_store)
