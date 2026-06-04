from graph_builder import kg

def find_closest_entity(query: str) -> str:
    """
    Finds the closest matching node in the graph.
    Handles cases like 'Vanshika' matching 'Vanshika Manjani'.
    """
    query_lower = query.lower()
    
    # Exact match first
    if query in kg.nodes:
        return query
    
    # Partial match — find nodes that CONTAIN the query word
    matches = []
    for node in kg.nodes:
        if query_lower in node.lower():
            matches.append(node)
    
    if matches:
        # Return shortest match (most specific)
        return min(matches, key=len)
    
    return None


def retrieve_graph_context(entity: str, max_depth: int = 2) -> str:
    """
    Starting from 'entity', walks the graph up to 'max_depth' hops.
    Now includes fuzzy matching so partial names work.
    """

    # Try to find closest entity if exact match fails
    actual_entity = find_closest_entity(entity)
    
    if not actual_entity:
        print(f"\n  Entity '{entity}' not found in graph.")
        print(f"  Available nodes: {list(kg.nodes())}\n")
        return ""
    
    if actual_entity != entity:
        print(f"\n  Matched '{entity}' → '{actual_entity}'")

    context = set()
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

    print(f"\n  Multi-hop context retrieved for '{actual_entity}' (depth={max_depth}):\n")
    for fact in context:
        print(f"  → {fact}")

    return ". ".join(context)