import networkx as nx
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOllama(model="qwen2.5:3b", temperature=0)

extract_prompt = PromptTemplate(
    template="""
You are a knowledge graph builder reading an academic research paper.
Extract ALL entities and relationships from the text below.

STRICT RULES:
- Return ONLY a valid JSON list. No explanation. No markdown.
- Each item must have exactly: "head", "relation", "tail"
- ONE entity per head, ONE entity per tail — never combine multiple people
- Extract people, institutions, concepts, tools, findings, scores
- For research findings use relations like: "found", "measured", "showed", "used", "studied"

GOOD EXAMPLES for academic text:
{{"head": "Vanshika Manjani", "relation": "studied at", "tail": "St. Xavier's College"}}
{{"head": "resilience", "relation": "is defined as", "tail": "positive adaptation despite adversity"}}
{{"head": "study", "relation": "used tool", "tail": "Bharathiar University Resilience Scale"}}
{{"head": "results", "relation": "showed", "tail": "no significant gender difference in resilience"}}
{{"head": "psychological well being", "relation": "includes", "tail": "autonomy"}}
{{"head": "psychological well being", "relation": "includes", "tail": "personal growth"}}

Text:
{text}

Output JSON:
""",
    input_variables=["text"],
)

extraction_chain = extract_prompt | llm | JsonOutputParser()

kg = nx.DiGraph()

def build_knowledge_graph(text: str):
    print("\n Sending text to Qwen 2.5 3B for triple extraction...\n")
    
    try:
        triples = extraction_chain.invoke({"text": text})
    except Exception as e:
        print(f" JSON parsing failed: {e}")
        print(" Try running again — small models sometimes return malformed JSON.")
        return kg

    # Filter out bad triples (missing keys or non-string values)
    clean_triples = []
    for t in triples:
        head = t.get("head", "")
        relation = t.get("relation", "")
        tail = t.get("tail", "")
        # Skip if any field is missing, empty, or not a string
        if not isinstance(head, str) or not isinstance(tail, str) or not isinstance(relation, str):
            continue
        if not head.strip() or not tail.strip() or not relation.strip():
            continue
        clean_triples.append({"head": head.strip(), "relation": relation.strip(), "tail": tail.strip()})

    print(f" Extracted {len(clean_triples)} valid triples:\n")
    for t in clean_triples:
        print(f"  {t['head']} --[{t['relation']}]--> {t['tail']}")

    print("\n Building Knowledge Graph...\n")
    for item in clean_triples:
        kg.add_node(item["head"])
        kg.add_node(item["tail"])
        kg.add_edge(item["head"], item["tail"], label=item["relation"])

    print(f" Graph built successfully!")
    print(f" Total Nodes : {kg.number_of_nodes()}")
    print(f" Total Edges : {kg.number_of_edges()}")
    return kg

def build_graph_from_chunks(chunks: list[str]):
    """
    Processes a list of text chunks one by one.
    Extracts triples from each chunk and adds them to the same graph.
    Useful for large PDFs where sending all text at once fails.
    """
    total_triples = 0

    for i, chunk in enumerate(chunks):
        print(f"\n Processing chunk {i+1}/{len(chunks)}...")

        try:
            triples = extraction_chain.invoke({"text": chunk})
        except Exception as e:
            print(f"  Skipping chunk {i+1} — JSON parse error: {e}")
            continue

        valid = 0
        for t in triples:
            head     = t.get("head", "")
            relation = t.get("relation", "")
            tail     = t.get("tail", "")

            if not all(isinstance(x, str) and x.strip() for x in [head, relation, tail]):
                continue

            kg.add_node(head.strip())
            kg.add_node(tail.strip())
            kg.add_edge(head.strip(), tail.strip(), label=relation.strip())
            valid += 1

        total_triples += valid
        print(f"  Added {valid} triples from chunk {i+1}")

    print(f"\n Total triples added: {total_triples}")
    print(f" Total Nodes : {kg.number_of_nodes()}")
    print(f" Total Edges : {kg.number_of_edges()}")
    return kg