# E2E RAG Pipeline Quest

## 1. Quick Start

To quickly get the pipeline running, ensure you have your environment variables set up and your vector database populated.

**1. Set up environment variables:**

Create a `.env` file in the root directory and add your OpenAI API key and model preferences:

```env
OPENAI_API_KEY=your_openai_api_key_here
LANGUAGE_MODEL=gpt-5-nano
EMBEDDING_MODEL=text-embedding-3-small
```

**2. Index the Data:**

Before running any queries, you need to populate the ChromaDB vector database with the sample data.

```bash
python indexer.py
```

**3. Run a Pipeline:**

Execute the basic RAG pipeline to test the setup:

```bash
python pipeline_1.py
```

---

## 2. Description

This repository contains a comprehensive collection of End-to-End Retrieval-Augmented Generation (RAG) pipelines using GL SDK. The project is structured into modular components (retrievers, response synthesizers, and handlers) which are then composed into seven distinct, executable pipeline examples. This allows for easy experimentation with different RAG optimization techniques.

---

## 3. Installation

**1. Clone the repository and navigate to the project directory:**

```bash
cd e2e-rag-pipeline-quest
```

**2. Create a virtual environment using **uv** (recommended):**

```bash
uv venv
source ~/venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**3. Install the required dependencies:**

Ensure you install the core pipeline libraries:

```bash
uv pip install --extra-index-url "https://oauth2accesstoken:$(gcloud auth print-access-token)@glsdk.gdplabs.id/gen-ai-internal/simple/" python-dotenv gllm-core gllm-generation gllm-inference gllm-pipeline gllm-retrieval gllm-datastore
```

---

## 4. Pipeline Architectures Overview

The repository includes 7 progressively advanced pipeline examples:

| Pipeline | Name | Description |
|---|---|---|
| `pipeline_1.py` | **Regular RAG** | A standard linear pipeline that retrieves context from ChromaDB and synthesizes a response. |
| `pipeline_2.py` | **Dynamic Step** | Introduces conditional execution (toggles) to dynamically bypass or trigger the retrieval step based on the pipeline state. |
| `pipeline_3.py` | **Semantic Routing** | Utilizes `semantic_router` to classify user intent, routing questions between a specific knowledge base branch and a general knowledge LLM branch (configured via `route_examples.json`). |
| `pipeline_4.py` | **Document References** | Implements a similarity-based reference formatter to explicitly cite and attach source chunks used in the final response. |
| `pipeline_5.py` | **Guardrails** | Integrates a guardrail manager with phrase-matching engines to intercept and block unsafe or banned queries before they reach the retriever. |
| `pipeline_6.py` | **Query Transformation** | Uses an LLM-based One-to-One Query Transformer to rewrite and optimize the user's initial prompt for better vector retrieval. |
| `pipeline_7.py` | **Multimodal Handling** | Demonstrates image processing capabilities by formatting image attachments (e.g., `dog1.png`, `dog2.png`) and passing them to a multimodal LLM for context-aware visual synthesis. |
| `pipeline_8.py` | **Caching** | Integrates a cache for eliminating redundant behavior in linear pipeline. |
| `pipeline_9.py` | **RAG with Dynamic Models** | A standard linear pipeline where the model is dynamic / not fixed. |
| `pipeline_10.py` | **Subgraphs** | Compose complex workflows by nesting smaller, reusable pipeline subgraphs within a larger parent pipeline. |
| `pipeline_11.py` | **Parallel Pipeline Processing** | Parallel execution of multiple independent steps (e.g., retrieving from multiple sources or querying models simultaneously) to improve latency. |
| `pipeline_12.py` | **Pipeline Step Exclusion** | Selectively omit or bypass certain predefined steps in a pipeline during execution based on runtime configuration. |

---

## 5. Project Structure

```
e2e-rag-pipeline-quest/
├── modules/
│   ├── retriever.py
│   ├── response_synthesizer.py
│   ├── semantic_router.py
│   └── handlers.py
├── pipelines/
│   ├── pipeline_1.py
│   ├── pipeline_2.py
    ...
│   ├── pipeline_9.py
│   └── pipeline_12.py
├── indexer.py
├── route_examples.json
└── .env
```

- **`/modules`** — Contains reusable components (`retriever.py`, `response_synthesizer.py`, `semantic_router.py`, `handlers.py`).
- **`/pipelines`** — Contains the executable pipeline scripts demonstrating various RAG patterns and architectures.
- **`indexer.py`** — The ingestion script responsible for loading raw CSV data, generating embeddings, and persisting them to a local ChromaDB instance.
- **`route_examples.json`** — Defines the utterance examples used to train the semantic router's intents.