import asyncio
import csv
from dotenv import load_dotenv
from gllm_core.schema import Chunk
from gllm_datastore.data_store import ChromaDataStore
from gllm_datastore.data_store.chroma.data_store import ChromaClientType
from gllm_inference.em_invoker import OpenAIEMInvoker

load_dotenv()

# Initialize data store with persistent storage and vector capability
em_invoker = OpenAIEMInvoker(model_name="text-embedding-3-small")
data_store = ChromaDataStore(
    collection_name="documents",
    client_type=ChromaClientType.PERSISTENT,  # use a Persistent Chroma DB
    persist_directory="data",  # 👈 where the data is located
).with_vector(em_invoker=em_invoker)


# Load documents from CSV file
async def load_csv_data():
    with open("data/imaginary_animals.csv", "r") as f:
        reader = csv.DictReader(f)
        chunks = [
            Chunk(content=row["description"], metadata={"name": row["name"]})
            for row in reader
        ]

    await data_store.vector.create(chunks)
    print(f"Successfully indexed {len(chunks)} documents from CSV file")


if __name__ == "__main__":
    asyncio.run(load_csv_data())
