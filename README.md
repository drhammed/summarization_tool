---
title: res-sum Research Evidence Synthesis
emoji: 📚
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
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

Hugging Face serves a free **Static Space** containing `index.html`, which
embeds the existing public app at `https://sum-tool.streamlit.app/?embed=true`.
The Python app continues running on Streamlit Community Cloud. Hugging Face
hosts the page around the app; it does not run Python or process uploaded PDFs.

Static Spaces do not require PRO. New Spaces no longer accept the native
Streamlit SDK, and new Docker or CPU Gradio Spaces require a paid plan. See
[Static HTML Spaces](https://huggingface.co/docs/hub/spaces-sdks-static) and
[Streamlit embedding](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app/embed-your-app).

GitHub remains the source of truth:

```text
Python app changes → GitHub main → Streamlit redeploys the app
Embed page changes → GitHub main → GitHub Actions updates the Hugging Face page
```

### One-time setup

1. Create a [Hugging Face access token](https://huggingface.co/settings/tokens)
   with write access to the target Space. The workflow can create
   `drhammed/summarization_tool` automatically if the token also permits creating
   repositories. The workflow creates a **public Static Space** by default.
   Alternatively, create the target Space first with the **Static** SDK and
   grant the token write access to it.
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
5. After the first successful sync, open
   `https://huggingface.co/spaces/drhammed/summarization_tool` (or your chosen
   Space). The page embeds the running Streamlit app and includes an
   **Open in new tab** link for direct access.

Later pushes to `main` sync automatically. Saving a file locally does not deploy
it; commit and push it to GitHub. Keep Streamlit Community Cloud connected to
this repository's `main` branch with `main.py` as its entry point. Python changes
then appear in the embedded app after Streamlit redeploys it.

The workflow uploads only `README.md` and `index.html` to Hugging Face. Updates
are one-way: edit these files in GitHub, because each sync replaces the Space's
files with this upload set. If you add separate CSS, JavaScript, or image assets,
add them to the workflow's `include` list too.

The workflow uses Hugging Face's official
[`hub-sync` action](https://huggingface.co/docs/hub/spaces-github-actions).
The root `sum-app.yml` is an older workflow example; GitHub only runs workflows
under `.github/workflows/`.

### API keys and session data

Visitors enter their LLM API key in the embedded app's sidebar, just as they do
when opening the Streamlit URL directly. Any operator-funded default API keys
belong on the **Streamlit backend**, where the Python app reads its environment
variables. Hugging Face Static Space variables and secrets are not passed to
the separately hosted Streamlit app. Keep credentials out of `index.html`.

Use a cloud LLM provider in the hosted app. The sidebar's local `ollama` option expects
an Ollama server on the hosting machine; the hosted app does not include one.

Uploaded PDFs, knowledge bases, and summaries stay in the Streamlit session's
temporary directories and can be lost when the Streamlit backend restarts or
redeploys. Keep the Streamlit app public and running so that it can be embedded.
If the embedded app is unavailable or asleep, use **Open in new tab** to access
the Streamlit app directly.

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

**Paper chunks per summary** controls how many paper passages are requested for
each summary, not how many passages are extracted or stored. A paper can produce
18 stored chunks while the slider requests 10 for its summary. If fewer chunks
are available, fewer are used. Global retrieval can add community summaries
alongside those paper chunks; Hybrid mode caps its combined context at the
selected count. Click **Run Analysis** again after changing the setting.
The Vector Store shows stored chunks, and Semantic Search has its own independent
result limit.

The **Chain-of-Thought Prompting** checkbox controls the instructions used for
summarization:

- **Checked:** combine the selected domain's built-in structured template with
  the text-box instructions. If the box is empty, use the domain template alone.
- **Unchecked:** use only the text-box instructions. If the box is empty, use
  the package's basic summarization prompt.

Whitespace-only text is treated as empty. The text-box instructions also guide
retrieval; the checkbox applies to summarization, not knowledge graph extraction.

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
