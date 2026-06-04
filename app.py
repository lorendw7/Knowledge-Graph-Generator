"""
============================================================================
 Streamlit Knowledge Graph Generator  --  heavily commented teaching version
============================================================================

This file wraps the logic from knowledge_graph.ipynb into a click-and-go web
app that runs in the browser.

How to run (from the project directory):
    streamlit run app.py
Your browser then opens http://localhost:8501 automatically.

----------------------------------------------------------------------------
 The single most important Streamlit concept: the "rerun model"
----------------------------------------------------------------------------
 A normal Python script runs top-to-bottom once and exits.
 Streamlit is different: every time the user interacts with the page (clicks a
 button, drags a slider, edits an input), Streamlit RE-RUNS THE ENTIRE app.py
 from the first line to the last, then redraws the page based on the st.xxx()
 calls it hits during that run.

 Consequences (the whole file is written around them):
   - The order you write code == the top-to-bottom order widgets appear.
   - Plain variables are reset on every rerun (so we use st.session_state to
     "remember" things across reruns).
   - Expensive work (like calling an LLM) must not happen on every rerun, so we
     gate it behind a button and cache the LLM client.
============================================================================
"""

# ---- standard library ----
import os        # read/write environment variables (OPENAI_API_KEY)
import asyncio   # LangChain's extraction call is an async coroutine; we await it via asyncio

# ---- third-party ----
import streamlit as st                  # Streamlit itself, conventionally aliased st
from dotenv import load_dotenv          # load environment variables from a .env file
from langchain_experimental.graph_transformers import LLMGraphTransformer  # text -> graph
from langchain_core.documents import Document       # LangChain "Document" object
from langchain_openai import ChatOpenAI             # call OpenAI chat models
from pyvis.network import Network                   # render the graph to interactive HTML

# load_dotenv() reads the project-root .env file and injects lines like
# "OPENAI_API_KEY=sk-..." into os.environ, so os.getenv() can read them below.
load_dotenv()


# ===========================================================================
# 1) Page configuration
# ---------------------------------------------------------------------------
# st.set_page_config MUST be the first st command in the script, otherwise
# Streamlit raises an error. It sets the browser tab title/icon and the layout.
#   layout="wide" -> use the full browser width (default is a narrow center column)
# ===========================================================================
st.set_page_config(page_title="Knowledge Graph Generator", page_icon="🕸️", layout="wide")

# st.title draws the largest heading at the top of the page.
st.title("🕸️ LLM Knowledge Graph Generator")
# st.caption draws a small grey line of text, typically used as a subtitle.
st.caption("Enter some text, extract entities & relationships with GPT, and visualize them interactively.")


# ===========================================================================
# 2) Sidebar (the left-hand column, used for configuration)
# ---------------------------------------------------------------------------
# Two ways to put widgets in the sidebar:
#   - st.sidebar.text_input(...)   -> prefix every widget with sidebar
#   - with st.sidebar:             -> a with-block; every st.xxx inside goes there
# We use the with-block form because it reads more cleanly.
# ===========================================================================
with st.sidebar:
    st.header("⚙️ Configuration")  # a medium heading inside the sidebar

    # st.text_input: single-line text box, returns the string the user typed.
    #   value=...        -> the default value (here, the key from .env if present)
    #   type="password"  -> mask the input as dots so the key isn't shoulder-surfed
    #   help=...         -> tooltip shown when hovering the little "?" icon
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),  # 2nd arg is the fallback if not set
        type="password",
        help="Leave blank to use OPENAI_API_KEY from .env",
    )

    # st.selectbox: a dropdown; returns the selected string.
    #   index=0 means the 0th item (gpt-4o) is selected by default.
    model_name = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1"], index=0)

    st.divider()  # a horizontal rule, purely decorative, to separate sections
    st.subheader("Schema constraints (optional)")

    # st.checkbox: returns True when ticked, False otherwise.
    use_node_schema = st.checkbox("Restrict node types", value=True)

    # Start with an empty list. Only show the multiselect if the box above is ticked.
    # This shows off the rerun model: a plain `if` can dynamically add/remove widgets.
    allowed_nodes = []
    if use_node_schema:
        # st.multiselect: returns a LIST of all selected items.
        #   default=...  -> items selected initially
        allowed_nodes = st.multiselect(
            "Allowed node types",
            ["Person", "Organization", "Location", "Award", "ResearchField", "Event"],
            default=["Person", "Organization", "Location", "ResearchField"],
        )


# ===========================================================================
# 3) Helper functions (with caching)
# ===========================================================================

# @st.cache_resource is a decorator placed above a function.
# Effect: it caches the function's return value. The next time the function is
# called with the SAME arguments, the body is NOT re-executed; the cached result
# is returned instead.
#
# Why we need it: the script reruns constantly. Without caching, every rerun
# would build a brand-new ChatOpenAI client -- slow and wasteful.
#
# cache_resource vs cache_data:
#   - cache_resource: for "resources/connections" that shouldn't be copied
#     (DB connections, model clients).
#   - cache_data: for "data" (DataFrames, lists, LLM responses).
#
# NOTE: a cached function's arguments must be HASHABLE, which is why the caller
# converts the `allowed_nodes` list into a tuple before passing it in.
@st.cache_resource(show_spinner=False)
def build_transformer(api_key: str, model_name: str, allowed_nodes: tuple):
    # Write the key back to the environment; ChatOpenAI reads it from there.
    os.environ["OPENAI_API_KEY"] = api_key
    # temperature=0 -> stable, reproducible output (same input -> same output).
    llm = ChatOpenAI(temperature=0, model_name=model_name)

    kwargs = {}
    if allowed_nodes:  # if the user restricted node types, pass them as a constraint
        kwargs["allowed_nodes"] = list(allowed_nodes)  # tuple back to list for LangChain
    # LLMGraphTransformer is the "text -> (nodes, relationships)" extractor.
    return LLMGraphTransformer(llm=llm, **kwargs)


def extract_graph(transformer, text: str):
    """Call the LLM to extract a knowledge graph from a piece of text."""
    # LangChain expects a list of Documents, not a raw string, so we wrap it.
    documents = [Document(page_content=text)]
    # aconvert_to_graph_documents is an async coroutine. In a notebook you can
    # `await` it directly, but in a plain .py script you can't, so we use
    # asyncio.run(...) to run the coroutine to completion and get the result.
    # The return value is a list, one GraphDocument per input Document.
    return asyncio.run(transformer.aconvert_to_graph_documents(documents))


def build_pyvis_html(graph_document) -> str:
    """Render a single GraphDocument into an interactive pyvis HTML string."""
    # New canvas. directed=True draws arrows on edges (relationships have direction).
    net = Network(height="600px", width="100%", directed=True,
                  bgcolor="#222222", font_color="white")

    nodes = graph_document.nodes              # all entity nodes
    relationships = graph_document.relationships  # all relationships (edges)
    # Build an id -> node dict to quickly check whether an id actually exists.
    node_dict = {n.id: n for n in nodes}

    # Add nodes one by one. group=n.type makes pyvis auto-color them by type.
    for n in nodes:
        net.add_node(n.id, label=n.id, title=n.type, group=n.type)

    # Add edges. Keep only those whose BOTH endpoints exist, to defend against
    # the LLM occasionally referencing a node it didn't list in `nodes`.
    for rel in relationships:
        if rel.source.id in node_dict and rel.target.id in node_dict:
            net.add_edge(rel.source.id, rel.target.id, label=rel.type.lower())

    # Configure the force-directed layout so the graph spreads out nicely.
    net.set_options("""
    {"physics": {"forceAtlas2Based": {"gravitationalConstant": -100,
     "centralGravity": 0.01, "springLength": 200, "springConstant": 0.08},
     "minVelocity": 0.75, "solver": "forceAtlas2Based"}}
    """)
    # generate_html() returns a complete HTML string (without writing a file),
    # ready to embed into the page via st.components.v1.html below.
    return net.generate_html(notebook=False)


# ===========================================================================
# 4) Main area: text input + trigger button
# ===========================================================================

# A default example so the user doesn't have to type something every time.
DEFAULT_TEXT = (
"""
Kyushu University (九州大学, Kyūshū Daigaku), abbreviated to Kyudai (九大, Kyūdai), is a public research university located in Fukuoka, Japan, on the island of Kyushu. Founded in 1911 as the fourth Imperial University in Japan, it has been recognised as a leading institution of higher education and research in Kyushu, Japan, and beyond.

The history of the university began a few decades before its founding when the medical school of the Fukuoka Domain (福岡藩) was established in 1867, the final year of the Edo period. The school was reorganised as the Fukuoka Medical College of Kyoto Imperial University in 1903. It became independent as Kyushu Imperial University in 1911.

History
In 1867, the Fukuoka Domain established a medical school called Sanshikan in Tenjin, Fukuoka. Although closed in 1872, its affiliated hospital continued operating. By 1879, it became part of the Fukuoka Prefectural Fukuoka Medical School, later continuing as the Fukuoka Prefectural Fukuoka Hospital.[5][failed verification]

The push for an imperial university in Kyushu led to the establishment of Fukuoka Medical College in 1903 as a branch of Kyoto Imperial University. Financial challenges delayed further development until the Furukawa Zaibatsu's donation in 1906 facilitated the establishment of Kyushu Imperial University in 1911, with Kenjiro Yamakawa, former president of the University of Tokyo, as its first president.[6][failed verification] In 1947, it was renamed Kyushu University, and in 1949, it expanded by incorporating several local educational institutions.[6][failed verification]

In 2003, the university was integrated with the Kyushu Institute of Design.[7] It opened its Ito campus in 2005.[6]

Organisation
Kyushu University's incumbent president is Tatsuro Ishibashi, who was elected in 2020 and is expected to serve until September 2026.[8]

The university has 16 faculties, 11 undergraduate schools, and 18 graduate schools.[9][10]

Faculty of Humanities
Faculty of Social and Cultural Studies
Faculty of Human-Environment Studies
Faculty of Law
Faculty of Economics
Faculty of Languages and Cultures
Faculty of Sciences
Faculty of Mathematics
Faculty of Medical Sciences
Faculty of Dental Science
Faculty of Pharmaceutical Sciences
Faculty of Engineering
Faculty of Engineering Sciences
Faculty of Design
Faculty of Information Science and Electrical Engineering
Faculty of Agriculture
Kyushu University Hospital
Kyushu University Hospital has historical roots in the 1867 Sanseikan established by the Kuroda Clan. Initially a clinic for a medical institution, it became affiliated with the Fukuoka Prefectural Medical School in 1879. In 1903, it became associated with the newly formed Fukuoka Medical College, a branch of Kyoto Imperial University. The establishment of Kyushu Imperial University in 1911 brought the hospital under its Faculty of Medicine. Post-World War II reforms in 1947 led to its rebranding as the Kyushu University Faculty of Medical Sciences Affiliated Hospital, incorporating various departments, including dental science. The hospital merged in 2003 with hospitals from the Faculty of Dental Science and Medical Institute of Bioregulation to form the current Kyushu University Hospital.[citation needed]

The hospital's history also includes fatal and torturous medical experiments including live dissection on American POWs by the university's medical faculty in 1945, resulting in war crimes convictions.
"""
)

# st.text_area: a multi-line text box, returns the string inside. height is in pixels.
text = st.text_area("📄 Input text", value=DEFAULT_TEXT, height=200)

# st.button: draws a button.
# Key point: it returns True ONLY on the single rerun where the user just clicked
# it, and False on every other rerun. So put "the action to perform" inside the if.
#   type="primary" -> the highlighted primary button (theme color)
if st.button("🚀 Generate knowledge graph", type="primary"):
    # Validate inputs one layer at a time, giving friendly feedback.
    if not api_key:
        # st.error: a red error banner
        st.error("Please provide an OpenAI API Key (sidebar or .env).")
    elif not text.strip():  # strip() removes surrounding whitespace to detect "empty"
        # st.warning: a yellow warning banner
        st.warning("Please enter some text.")
    else:
        # st.spinner: shows a spinner + message while the with-block runs, then
        # disappears. Ideal for wrapping slow operations (here, the LLM call).
        with st.spinner("Calling the LLM to extract entities & relationships..."):
            # Convert list -> tuple before passing to the cached function (hashable).
            transformer = build_transformer(api_key, model_name, tuple(allowed_nodes))
            graph_documents = extract_graph(transformer, text)

        # IMPORTANT: store the result in st.session_state.
        # session_state is a dict that persists across reruns -- like memory for
        # this browser session. If we only used a plain variable, the next rerun
        # (e.g. when the user toggles a checkbox) would lose it and the graph
        # would vanish.
        st.session_state["graph_doc"] = graph_documents[0]
        # st.success: a green success banner
        st.success("Extraction complete!")


# ===========================================================================
# 5) Render the result
# ---------------------------------------------------------------------------
# This block is NOT inside the button's if. It independently checks whether a
# result exists in session_state. That way the graph stays visible across any
# rerun once it has been extracted, instead of flashing away.
# ===========================================================================
if "graph_doc" in st.session_state:
    gdoc = st.session_state["graph_doc"]  # retrieve the previously stored graph

    # st.columns(2): split the page into 2 equal-width columns, returning 2
    # container objects. Calling st.xxx on col1/col2 places content in that column.
    col1, col2 = st.columns(2)
    # st.metric: shows a big number with a label, like a KPI card.
    col1.metric("Nodes", len(gdoc.nodes))
    col2.metric("Relationships", len(gdoc.relationships))

    # st.tabs: tabbed sections, returning a container per tab.
    tab_graph, tab_data = st.tabs(["🕸️ Graph", "📋 Raw data"])

    with tab_graph:
        html = build_pyvis_html(gdoc)  # build the pyvis HTML string
        # st.components.v1.html: embed arbitrary HTML into the page (via an iframe).
        # This is the general way to put third-party visualizations (pyvis / echarts
        # / d3 / ...) inside a Streamlit app.
        st.components.v1.html(html, height=620, scrolling=True)

    with tab_data:
        st.subheader("Nodes")
        # st.json: renders Python lists/dicts as a pretty, collapsible JSON view.
        st.json([{"id": n.id, "type": n.type} for n in gdoc.nodes])
        st.subheader("Relationships")
        st.json([
            {"source": r.source.id, "type": r.type, "target": r.target.id}
            for r in gdoc.relationships
        ])
