# res-sum Streamlit App

A web interface for [res-sum](https://pypi.org/project/res-sum/) — research paper summarization with GraphRAG.

## Features

- **Upload PDFs** and generate structured summaries using LLMs
- **Interactive Knowledge Graph** — visualize entities and relationships extracted from papers
- **Vector Store Browser** — search chunks by semantic similarity, browse by paper
- **Community Detection** — view topic clusters and their LLM-generated summaries
- **Multiple LLM providers** — Ollama Cloud, Groq, OpenAI, Anthropic

## Live Demo

[sum-tool.streamlit.app](https://sum-tool.streamlit.app/)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

## How It Works

This app is a thin UI layer over the `res-sum` Python package. All the heavy lifting (PDF parsing, knowledge graph construction, vector embeddings, hybrid retrieval, summarization) is handled by `res-sum`.

```
Upload PDFs → res-sum ingests → ChromaDB + NetworkX → Hybrid retrieval → LLM summary
```

## Configuration

Set API keys as environment variables to avoid re-entering them:

```bash
export OLLAMA_API_KEY="your-key"    # for Ollama Cloud
export GROQ_API_KEY="your-key"      # for Groq
export OPENAI_API_KEY="your-key"    # for OpenAI
export ANTHROPIC_API_KEY="your-key" # for Anthropic
```

## Powered By

- [res-sum](https://pypi.org/project/res-sum/) — GraphRAG-powered research synthesis
- [Streamlit](https://streamlit.io/) — Python web framework
