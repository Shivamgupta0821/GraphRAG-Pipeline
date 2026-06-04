import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import io
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

#Page config 
st.set_page_config(
    page_title="GraphRAG Chatbot",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS 
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stTextInput > div > div > input {
        background-color: #1e2130;
        color: white;
        border-radius: 8px;
    }
    .chat-message-user {
        background-color: #1e3a5f;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        color: white;
    }
    .chat-message-bot {
        background-color: #1e2130;
        padding: 12px 16px;
        border-radius: 12px;
        margin: 8px 0;
        color: #e0e0e0;
        border-left: 3px solid #4A90D9;
    }
    .fact-box {
        background-color: #0d1f0d;
        padding: 10px 14px;
        border-radius: 8px;
        border-left: 3px solid #4CAF50;
        font-size: 13px;
        color: #90EE90;
        margin: 4px 0;
    }
    .stat-box {
        background-color: #1e2130;
        padding: 16px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

#Session State 
if "kg"            not in st.session_state: st.session_state.kg            = nx.DiGraph()
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "graph_built"   not in st.session_state: st.session_state.graph_built   = False
if "total_triples" not in st.session_state: st.session_state.total_triples = 0

# LLM Setup 
@st.cache_resource
def load_llm():
    return ChatOllama(model="qwen2.5:3b", temperature=0)

llm = load_llm()

#Prompts 
extract_prompt = PromptTemplate(
    template="""
You are a knowledge graph builder.
Extract entities and relationships from the text below.

STRICT RULES:
- Return ONLY a valid JSON list. No explanation. No markdown.
- Each item must have exactly: "head", "relation", "tail"
- ONE entity per head, ONE entity per tail
- Keep entities short and clean

Text:
{text}

Output JSON:
""",
    input_variables=["text"],
)

final_prompt = PromptTemplate(
    template="""
Answer the question using ONLY the context provided below.
Do not use any outside knowledge.
If the answer is not in the context, say "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

from langchain_core.output_parsers import JsonOutputParser
extraction_chain = extract_prompt | llm | JsonOutputParser()
rag_chain        = final_prompt   | llm

#Helper Functions 
def clean_text(text: str) -> str:
    lines   = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) < 20:
            continue
        if sum(c.isdigit() for c in line) > len(line) * 0.5:
            continue
        cleaned.append(line)
    return " ".join(cleaned)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    if not text.strip():
        return []
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            last_period = text.rfind(".", start, end)
            if last_period != -1 and last_period > start + 50:
                end = last_period + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def build_graph_from_chunks(chunks: list, progress_bar, status_text):
    kg             = nx.DiGraph()
    total_triples  = 0

    for i, chunk in enumerate(chunks):
        progress = (i + 1) / len(chunks)
        progress_bar.progress(progress)
        status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")

        try:
            triples = extraction_chain.invoke({"text": chunk})
        except Exception:
            continue

        # Flatten if model returned nested list
        if triples and isinstance(triples[0], list):
            triples = [item for sublist in triples for item in sublist]

        for t in triples:
            # Skip if not a dict
            if not isinstance(t, dict):
                continue

            head     = t.get("head",     "")
            relation = t.get("relation", "")
            tail     = t.get("tail",     "")

            if not all(isinstance(x, str) and x.strip()
                       for x in [head, relation, tail]):
                continue

            kg.add_node(head.strip())
            kg.add_node(tail.strip())
            kg.add_edge(head.strip(), tail.strip(), label=relation.strip())
            total_triples += 1

    return kg, total_triples


def find_closest_entity(query: str, kg) -> str:
    if query in kg.nodes:
        return query
    query_lower = query.lower()
    matches     = [n for n in kg.nodes if query_lower in n.lower()]
    if matches:
        return min(matches, key=len)
    return None


def retrieve_graph_context(entity: str, kg, max_depth: int = 3):
    actual_entity = find_closest_entity(entity, kg)
    if not actual_entity:
        return "", [], None

    context       = set()
    visited_nodes = set()

    def dfs(node, depth):
        if depth > max_depth:
            return
        visited_nodes.add(node)
        for neighbor in kg.successors(node):
            relation = kg.get_edge_data(node, neighbor)["label"]
            context.add(f"{node} {relation} {neighbor}")
            if neighbor not in visited_nodes:
                dfs(neighbor, depth + 1)
        for predecessor in kg.predecessors(node):
            relation = kg.get_edge_data(predecessor, node)["label"]
            context.add(f"{predecessor} {relation} {node}")
            if predecessor not in visited_nodes:
                dfs(predecessor, depth + 1)

    dfs(actual_entity, 1)
    return ". ".join(context), list(context), actual_entity


def render_graph(kg, highlight_nodes=None):
    if kg.number_of_nodes() == 0:
        return None

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    pos          = nx.spring_layout(kg, seed=42, k=2.5)
    node_degrees = dict(kg.degree())
    hub_nodes    = [n for n, d in node_degrees.items() if d >= 3]
    normal_nodes = [n for n, d in node_degrees.items() if d <  3]

    # Highlighted nodes (used in last answer)
    if highlight_nodes:
        highlighted = [n for n in kg.nodes if n in highlight_nodes]
    else:
        highlighted = []

    nx.draw_networkx_nodes(kg, pos, nodelist=hub_nodes,
        node_color="#4A90D9", node_size=2000, alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(kg, pos, nodelist=normal_nodes,
        node_color="#2d4a2d", node_size=1000, alpha=0.8, ax=ax)
    if highlighted:
        nx.draw_networkx_nodes(kg, pos, nodelist=highlighted,
            node_color="#FFD700", node_size=2200, alpha=1.0, ax=ax)

    nx.draw_networkx_edges(kg, pos,
        edge_color="#444444", arrows=True,
        arrowsize=15, width=1.2,
        connectionstyle="arc3,rad=0.1", ax=ax)

    nx.draw_networkx_labels(kg, pos,
        font_size=7, font_color="white",
        font_weight="bold", ax=ax)

    edge_labels = nx.get_edge_attributes(kg, "label")
    nx.draw_networkx_edge_labels(kg, pos,
        edge_labels=edge_labels,
        font_size=5, font_color="#FF8C00",
        label_pos=0.35, ax=ax)

    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor="#0e1117")
    buf.seek(0)
    plt.close(fig)
    return buf


#UI LAYOUT

st.title("🧠 GraphRAG Chatbot")
st.caption("Upload a PDF → Build a Knowledge Graph → Ask questions with Multi-Hop Reasoning")

# Sidebar 
with st.sidebar:
    st.header("📁 Upload Document")

    upload_mode = st.radio(
        "Data Source",
        ["Upload PDF", "Use sample.txt"],
        index=0
    )

    if upload_mode == "Upload PDF":
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    else:
        uploaded_file = None

    build_btn = st.button("🔨 Build Knowledge Graph", use_container_width=True)

    st.divider()

    if st.session_state.graph_built:
        st.markdown("### 📊 Graph Stats")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Nodes", st.session_state.kg.number_of_nodes())
        with col2:
            st.metric("Edges", st.session_state.kg.number_of_edges())

        st.metric("Triples Extracted", st.session_state.total_triples)

        st.divider()
        st.markdown("### 🔍 All Entities")
        nodes_list = sorted(list(st.session_state.kg.nodes()))
        for node in nodes_list:
            st.caption(f"• {node}")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ── Build Graph Logic ─────────────────────────────────────────
if build_btn:
    text = ""

    if upload_mode == "Upload PDF" and uploaded_file:
        from pypdf import PdfReader
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        reader = PdfReader(tmp_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        os.unlink(tmp_path)

    elif upload_mode == "Use sample.txt":
        try:
            with open("data/sample.txt", "r") as f:
                text = f.read()
        except FileNotFoundError:
            st.error("data/sample.txt not found.")

    else:
        st.warning("Please upload a PDF first.")

    if text.strip():
        cleaned = clean_text(text)
        chunks  = chunk_text(cleaned, chunk_size=300, overlap=50)

        st.info(f"📄 Text extracted. Split into {len(chunks)} chunks. Building graph...")

        progress_bar = st.progress(0)
        status_text  = st.empty()

        kg, total = build_graph_from_chunks(chunks, progress_bar, status_text)

        st.session_state.kg            = kg
        st.session_state.total_triples = total
        st.session_state.graph_built   = True

        status_text.text("✅ Graph built successfully!")
        progress_bar.progress(1.0)
        st.success(f"Graph ready! {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges.")
        st.rerun()

# ── Main Area ─────────────────────────────────────────────────
tab1, tab2 = st.tabs(["💬 Chat", "🕸️ Knowledge Graph"])

with tab1:
    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-message-user">🧑 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message-bot">🤖 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
                if msg.get("facts"):
                    with st.expander("🔗 Graph facts used"):
                        for fact in msg["facts"]:
                            st.markdown(
                                f'<div class="fact-box">→ {fact}</div>',
                                unsafe_allow_html=True
                            )

    st.divider()

    # Input area
    if not st.session_state.graph_built:
        st.info("👈 Build a Knowledge Graph first using the sidebar.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            entity = st.text_input(
                "Entity",
                placeholder="e.g. Resilience",
                key="entity_input"
            )
        with col2:
            question = st.text_input(
                "Question",
                placeholder="e.g. What is resilience?",
                key="question_input"
            )

        ask_btn = st.button("🚀 Ask", use_container_width=False)

        if ask_btn and entity and question:
            context, facts, matched_entity = retrieve_graph_context(
                entity, st.session_state.kg, max_depth=3
            )

            if not context:
                st.warning(f"Entity '{entity}' not found. Check the entities list in the sidebar.")
            else:
                if matched_entity != entity:
                    st.caption(f"Matched '{entity}' → '{matched_entity}'")

                with st.spinner("Thinking..."):
                    response = rag_chain.invoke({
                        "context": context,
                        "question": question
                    })
                    answer = response.content

                # Save to chat history
                st.session_state.chat_history.append({
                    "role"   : "user",
                    "content": f"[{entity}] {question}"
                })
                st.session_state.chat_history.append({
                    "role"   : "bot",
                    "content": answer,
                    "facts"  : facts
                })

                st.rerun()

with tab2:
    if not st.session_state.graph_built:
        st.info("👈 Build a Knowledge Graph first.")
    else:
        st.subheader("🕸️ Knowledge Graph Visualization")

        buf = render_graph(st.session_state.kg)
        if buf:
            st.image(buf, use_container_width=True)

        st.download_button(
            label     = "⬇️ Download Graph Image",
            data      = buf,
            file_name = "knowledge_graph.png",
            mime      = "image/png"
        )