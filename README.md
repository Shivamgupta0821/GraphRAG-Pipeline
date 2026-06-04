# 🧠 GraphRAG Pipeline — Knowledge Graph + Multi-Hop Reasoning

A production-style GraphRAG pipeline that extracts knowledge from PDFs, builds a Knowledge Graph, and answers complex multi-hop questions using local LLMs. Runs 100% locally — no API keys, no cost.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python) ![LangChain](https://img.shields.io/badge/LangChain-green?style=flat-square) ![Streamlit](https://img.shields.io/badge/Streamlit-red?style=flat-square&logo=streamlit) ![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=flat-square) ![NetworkX](https://img.shields.io/badge/NetworkX-orange?style=flat-square)

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Example Questions](#example-questions)
- [Real World Applications](#real-world-applications)
- [Author](#author)

---

## 🌟 Overview

Standard RAG (Retrieval Augmented Generation) breaks documents into chunks and searches by similarity. It finds isolated facts but **misses connections between them**.

**GraphRAG fixes this** — it stores knowledge as a network of connected facts and traverses relationships to answer questions that span multiple sentences or documents. This is called **Multi-Hop Reasoning**.

**Example:**
Question: "Which company invested in the company that built ChatGPT?"
Standard RAG returns: "ChatGPT is used by millions of users"  ❌
GraphRAG traverses:
ChatGPT → powered by → GPT-4
GPT-4   → developed by → OpenAI
OpenAI  → invested in by → Microsoft
Answer: Microsoft ✅

---

## ✨ Features

- 📄 Upload any PDF and auto-build a Knowledge Graph
- 🕸️ Interactive graph visualization with color-coded nodes
- 🔗 Multi-hop reasoning across connected facts
- 💬 Chat interface with full conversation history
- 🔍 Fuzzy entity matching — type "Vanshika" to find "Vanshika Manjani"
- 📊 Real-time progress bar during graph construction
- ⬇️ Download the knowledge graph as an image
- 🖥️ Runs 100% locally — no OpenAI API key, no cost

---

## 🛠 Tech Stack

### Core
| Technology | Purpose |
|------------|---------|
| Ollama + Qwen 2.5 3B | Local LLM for triple extraction and answering |
| LangChain | LLM orchestration and prompt management |
| NetworkX | Knowledge graph construction and traversal |

### Interface
| Technology | Purpose |
|------------|---------|
| Streamlit | Web UI and chat interface |
| matplotlib | Graph visualization |

### Data
| Technology | Purpose |
|------------|---------|
| pypdf | PDF text extraction |
| python-dotenv | Environment management |

---

## 📁 Project Structure
GraphRAG-Pipeline/
│
├── streamlit_app.py      # Main Streamlit UI
├── graph_builder.py      # Triple extraction + graph construction
├── retriever.py          # Multi-hop graph traversal
├── visualizer.py         # Graph visualization
├── pdf_loader.py         # PDF text extraction
├── chunker.py            # Text chunking with overlap
├── app.py                # Terminal version
├── requirements.txt      # Dependencies
│
├── data/
│   └── sample.txt        # Sample knowledge base
│
├── pdfs/                 # Drop your PDFs here
│
└── graphs/
└── knowledge_graph.png

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed

### 1. Clone the repository
```bash
git clone https://github.com/Shivamgupta0821/GraphRAG-Pipeline.git
cd GraphRAG-Pipeline
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull the local LLM
```bash
ollama pull qwen2.5:3b
```

### 5. Run the app
```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## 🎯 How It Works

1. Upload any PDF using the sidebar
2. Click **Build Knowledge Graph** — the LLM reads every chunk and extracts entity-relationship triples
3. The triples are stored as a directed graph using NetworkX
4. Type an entity and a question in the Chat tab
5. The system walks the graph up to 3 hops from your entity, collecting connected facts
6. Those facts are sent to the LLM as context to generate the final answer
7. View the full Knowledge Graph in the Graph tab and download it

---

## 🧪 Example Questions

| Entity | Question | Hops |
|--------|----------|------|
| `ChatGPT` | Which company invested in the company that built ChatGPT? | 3 |
| `YouTube` | Who founded the company that acquired YouTube? | 2 |
| `Resilience` | What factors affect resilience in individuals? | 1 |
| `WHO` | How did WHO define health? | 1 |
| `results` | What did the results show about gender differences? | 2 |
| `Vanshika` | Which college does Vanshika study in? | 1 |

---

## 💡 Real World Applications

| Domain | Use Case |
|--------|----------|
| 🏥 Healthcare | Connect drugs → proteins → diseases across research papers |
| 🏦 Finance | Map company relationships, investments and acquisitions |
| ⚖️ Legal | Link entities, clauses and precedents across case documents |
| 🔍 Fraud Detection | Find hidden connections between suspicious entities |
| 📚 Research | Synthesize findings across hundreds of academic papers |
| 🏭 Supply Chain | Map supplier relationships and dependencies |

---

## 🗺️ Roadmap

- [x] PDF ingestion pipeline
- [x] LLM-powered triple extraction
- [x] Knowledge graph construction
- [x] Multi-hop graph traversal
- [x] Streamlit chat UI
- [x] Graph visualization
- [x] Fuzzy entity matching
- [ ] Vector + Graph hybrid search
- [ ] Multi-PDF support
- [ ] Export graph as JSON
- [ ] Docker support

---

## 👨‍💻 Author

**Shivam Gupta**

GitHub: [@Shivamgupta0821](https://github.com/Shivamgupta0821) · LinkedIn: https://www.linkedin.com/in/shivam-gupta-b03442289

---

## ⭐ If you found this useful, please star the repo!
