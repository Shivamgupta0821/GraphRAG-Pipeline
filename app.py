from graph_builder import build_knowledge_graph, build_graph_from_chunks
from retriever import retrieve_graph_context
from visualizer import visualize_graph
from pdf_loader import load_pdfs_from_folder, clean_text
from chunker import chunk_text
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
import os

llm = ChatOllama(model="qwen2.5:3b", temperature=0)

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

rag_chain = final_prompt | llm

print("\n" + "="*60)
print("   GraphRAG Pipeline — Powered by Qwen 2.5 3B + NetworkX")
print("="*60)

# ── Decide: use PDFs or fallback to sample.txt ───────────────
pdf_text = load_pdfs_from_folder("pdfs")

if pdf_text.strip():
    print("\n Using PDF data...\n")
    cleaned  = clean_text(pdf_text)
    chunks   = chunk_text(cleaned, chunk_size=300, overlap=50)
    print(f" Split into {len(chunks)} chunks\n")
    kg = build_graph_from_chunks(chunks)

else:
    print("\n No PDFs found — falling back to sample.txt\n")
    with open("data/sample.txt", "r") as f:
        text = f.read()
    kg = build_knowledge_graph(text)

# ── Ask your questions ────────────────────────────────────────
print("\n" + "="*60)
print("   What would you like to ask?")
print("="*60)
print("\n Type an entity to start from (e.g. ChatGPT, Microsoft)")
print(" Then type your question.")
print(" Type 'quit' to exit and see the graph.\n")

while True:
    entity = input(" Entity : ").strip()
    if entity.lower() == "quit":
        break

    question = input(" Question: ").strip()
    if question.lower() == "quit":
        break

    context = retrieve_graph_context(entity, max_depth=3)

    if not context:
        print(" Entity not found in graph. Try another.\n")
        continue

    response = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print(f"\n Answer: {response.content}\n")
    print("─"*60)

print("\n" + "="*60)
print("   Pipeline Complete — Generating Graph Visualization")
print("="*60 + "\n")

visualize_graph()