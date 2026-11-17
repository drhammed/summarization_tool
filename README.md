---
title: res-sum Research Evidence Synthesis
emoji: 📚
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.44.1"
python_version: "3.11"
app_file: main.py
pinned: false
---

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

## Deploy to Hugging Face Spaces

The app uses Hugging Face's native Streamlit SDK on the default CPU Basic
hardware, matching the SDK and hardware used by
[meta-analysis-extractor](https://huggingface.co/spaces/drhammed/meta-analysis-extractor).
The YAML metadata above selects Python 3.11, Streamlit 1.44.1, and `main.py` as
the entry point. Hugging Face installs Python dependencies from
`requirements.txt` and system dependencies from `packages.txt`.

The native Streamlit SDK is deprecated, although existing Spaces still run it.
This configuration tests whether Hugging Face still permits creating a new
native Streamlit Space for this account. Successful creation is not guaranteed:
the first sync may still be rejected by Hugging Face's hosting eligibility
rules. See the [Streamlit Spaces documentation](https://huggingface.co/docs/hub/spaces-sdks-streamlit).

GitHub remains the source of truth:

```text
Push to GitHub main → GitHub Actions syncs the app → Hugging Face rebuilds the Space
```

### One-time setup

1. Create a [Hugging Face access token](https://huggingface.co/settings/tokens)
   with write access to the target Space. The workflow can create
   `drhammed/summarization_tool` automatically if the token also permits creating
   repositories. The workflow requests the native Streamlit SDK through the API
   and creates a **public** Space by default. The current New Space form may
   not offer this deprecated SDK.
2. In this GitHub repository's
   [Actions secrets settings](https://github.com/drhammed/summarization_tool/settings/secrets/actions),
   add a repository secret named **`HF_TOKEN`** with that token as its value.
   This is a deployment token; keep it in GitHub, separate from the LLM API keys.
3. To use a different Space name, add a GitHub Actions repository **variable**
   named `HF_SPACE_ID`, such as `drhammed/res-sum`. Otherwise, the default is
   `drhammed/summarization_tool`.
4. Commit and push these deployment files to `main`. In the repository's
   **Actions** tab, open **Sync to Hugging Face Space** to follow the sync.
   You can also choose **Run workflow** on `main` to retry without a new commit.
5. After the first successful sync, follow the build logs at
   `https://huggingface.co/spaces/drhammed/summarization_tool` (or your chosen
   Space). The app becomes available after the build completes.

If creation returns `402 Payment Required`, the native SDK attempt is also
blocked by account eligibility. CPU Basic is a hardware tier; choosing it does
not guarantee that new Space creation is free. The
[current Spaces documentation](https://huggingface.co/docs/hub/spaces-overview)
describes the subscription requirements for new compute Spaces.

Later pushes to `main` sync automatically. Saving a file locally does not deploy
it; commit and push it to GitHub. Updates are one-way: edit the source in GitHub,
because the next sync replaces Space files with the GitHub version, including
removing files deleted from GitHub. The existing Streamlit Community Cloud app
can continue deploying from the same repository.

The workflow uses Hugging Face's official
[`hub-sync` action](https://huggingface.co/docs/hub/spaces-github-actions).
The root `sum-app.yml` is an older workflow example; GitHub only runs workflows
under `.github/workflows/`.

### API keys and session data

Visitors can enter their own LLM API key in the sidebar. To provide an
operator-funded default instead, add the relevant `OLLAMA_API_KEY`,
`GROQ_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY` under the Space's
**Settings → Variables and secrets → Secrets**. The existing app reads these
environment variables. A default key is available to everyone using the public
app, and usage is charged to its owner; leave it unset for visitor-provided keys.

Use a cloud LLM provider on Spaces. The sidebar's local `ollama` option expects
an Ollama server on the hosting machine; the hosted app does not include one.

Uploaded PDFs, knowledge bases, and summaries use temporary session directories;
they are lost on a Space restart or rebuild. Download results before redeploying.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

For local Docker development:

```bash
docker build -t res-sum-space .
docker run --rm -p 7860:7860 res-sum-space
```

Open `http://localhost:7860` and enter an API key in the sidebar.

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
