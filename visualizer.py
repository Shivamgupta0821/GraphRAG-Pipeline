import networkx as nx
import matplotlib.pyplot as plt
from graph_builder import kg

def visualize_graph(output_path: str = "graphs/knowledge_graph.png"):
    """
    Draws the knowledge graph and saves it as an image.
    """
    if kg.number_of_nodes() == 0:
        print(" Graph is empty. Build the graph first.")
        return

    print("\n Generating Knowledge Graph visualization...\n")

    plt.figure(figsize=(18, 12))
    plt.title("GraphRAG Knowledge Graph", fontsize=16, fontweight="bold", pad=20)

    # Layout — spring layout spreads nodes naturally
    pos = nx.spring_layout(kg, seed=42, k=2.5)

    # Separate nodes into "important" ones (high connections) and regular
    node_degrees = dict(kg.degree())
    hub_nodes    = [n for n, d in node_degrees.items() if d >= 3]
    normal_nodes = [n for n, d in node_degrees.items() if d < 3]

    # Draw hub nodes (companies, key people) larger + different color
    nx.draw_networkx_nodes(kg, pos,
        nodelist=hub_nodes,
        node_color="#4A90D9",
        node_size=2500,
        alpha=0.95)

    # Draw normal nodes smaller
    nx.draw_networkx_nodes(kg, pos,
        nodelist=normal_nodes,
        node_color="#A8D8A8",
        node_size=1200,
        alpha=0.85)

    # Draw edges (arrows)
    nx.draw_networkx_edges(kg, pos,
        edge_color="#888888",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        width=1.5,
        connectionstyle="arc3,rad=0.1")

    # Draw node labels
    nx.draw_networkx_labels(kg, pos,
        font_size=8,
        font_weight="bold",
        font_color="black")

    # Draw edge labels (the relationship names)
    edge_labels = nx.get_edge_attributes(kg, "label")
    nx.draw_networkx_edge_labels(kg, pos,
        edge_labels=edge_labels,
        font_size=6,
        font_color="#CC0000",
        label_pos=0.35)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    print(f" Graph saved to: {output_path}")
    plt.show()