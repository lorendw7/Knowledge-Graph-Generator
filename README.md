# Knowledge Graph Generator

Turn a paragraph of natural-language text into an interactive knowledge graph using an LLM (GPT-4o) and visualize it in the browser with [pyvis](https://pyvis.readthedocs.io/).

This repo demonstrates LangChain's `LLMGraphTransformer` with three levels of schema strictness:

1. **Unconstrained** — the LLM picks node and relationship types freely.
2. **Node-type whitelist** — only emit nodes of allowed types.
3. **Node + relationship whitelist** — also pin the relationship schema.

The included example extracts a graph from the English Wikipedia description of **Kyushu University**.

## Requirements

- Python 3.10+
- An OpenAI API key with access to `gpt-4o`

## Setup

```bash
# 1. Clone
git clone https://github.com/lorendw7/Knowledge-Graph-Generator.git
cd Knowledge-Graph-Generator

# 2. (Optional) create a virtualenv
python -m venv .venv
.venv\Scripts\activate     # Windows PowerShell
# source .venv/bin/activate  # macOS / Linux

# 3. Install dependencies (the first notebook cell does this for you, too)
pip install --upgrade langchain langchain-experimental langchain-openai python-dotenv pyvis
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
```

> `.env` is gitignored — never commit secrets.

## Running

Open `knowledge_graph.ipynb` in Jupyter / VS Code and run all cells. The notebook will:

1. Load the API key from `.env`.
2. Send the example text to GPT-4o via `LLMGraphTransformer`.
3. Print the extracted `Node` / `Relationship` lists.
4. Render `knowledge_graph.html` and open it in your default browser.

To use your own text, edit the `text = """ ... """` block in the cell under **"Extract graph data"**.

## How it works

| Stage | Notebook cell | What happens |
|---|---|---|
| Load secrets | `dotenv` cell | Read `OPENAI_API_KEY` from `.env` |
| Build pipeline | `ChatOpenAI` + `LLMGraphTransformer` | Wire GPT-4o into the graph extractor |
| Extract | `aconvert_to_graph_documents(documents)` | LLM returns structured `(node, relationship)` triples |
| Visualize | `visualize_graph(...)` | pyvis renders an interactive HTML graph with force-directed layout |

The `visualize_graph` helper defensively filters out edges whose endpoints aren't in the node list, then groups nodes by type so pyvis can color-code them.

## Schema constraints

Constraining the schema produces cleaner, more useful graphs:

```python
allowed_nodes = ["Person", "Organization", "Location", "Award", "ResearchField"]
allowed_relationships = [
    ("Organization", "LOCATED_IN", "Location"),
    ("Organization", "MERGED_WITH", "Organization"),
    ("Person",       "ALUMNUS_OF", "Organization"),
    # ...
]
```

Pass these into `LLMGraphTransformer(allowed_nodes=..., allowed_relationships=...)`.

## Output

`knowledge_graph.html` — a self-contained interactive graph. Drag nodes to rearrange, hover to see node types. This file is gitignored because it is regenerated on every run.

## License

MIT (see `LICENSE` if present).

---

中文版说明见 [README.zh.md](./README.zh.md)。
