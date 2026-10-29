import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. BUILD THE GRAPH
# ---------------------------------------------------------
# A MultiDiGraph lets us have multiple edge types between
# the same nodes (e.g., a fan can both ATTEND and be a MEMBER)

### Some Definitions & Examples
# Entities (nodes)	       Fans, teams, cities, venues, events
# Relationships (edges)	   FOLLOWS, LIVES_IN, PLAYS_AT, ATTENDS
# Ontology / schema	       The type attribute on every node
# Multi-hop traversal	Query 3 chains 3 edges: LIVES_IN → FOLLOWS → PLAYS_AT
# Graph reasoning	"Fans in LA whose teams play at SoFi" is not a SQL JOIN — it's a path pattern
# Relationship direction	Query 2 uses in_edges because ATTENDS points fan → event


G = nx.MultiDiGraph()

# --- Add nodes with types (like a schema) ---
fans = {
    "sarah":  {"type": "Fan", "name": "Sarah"},
    "mike":   {"type": "Fan", "name": "Mike"},
    "priya":  {"type": "Fan", "name": "Priya"},
}
teams = {
    "lafc":   {"type": "Team", "name": "LAFC"},
    "lagalaxy": {"type": "Team", "name": "LA Galaxy"},
    "sounders": {"type": "Team", "name": "Seattle Sounders"},
}
cities = {
    "la":     {"type": "City", "name": "Los Angeles"},
    "sea":    {"type": "City", "name": "Seattle"},
}
venues = {
    "bmo":    {"type": "Venue", "name": "BMO Stadium"},
    "sofi":   {"type": "Venue", "name": "SoFi Stadium"},
    "lumen":  {"type": "Venue", "name": "Lumen Field"},
}
events = {
    "el_trafico": {"type": "Event", "name": "El Tráfico, May 2024"},
    "cascadia":   {"type": "Event", "name": "Cascadia Cup, June 2024"},
}

for group in (fans, teams, cities, venues, events):
    for node_id, attrs in group.items():
        G.add_node(node_id, **attrs)

# --- Add relationships (edges) ---
# Fans live in cities
G.add_edge("sarah", "la",  rel="LIVES_IN")
G.add_edge("mike",  "la",  rel="LIVES_IN")
G.add_edge("priya", "sea", rel="LIVES_IN")

# Fans follow teams
G.add_edge("sarah", "lafc",     rel="FOLLOWS")
G.add_edge("mike",  "lagalaxy", rel="FOLLOWS")
G.add_edge("priya", "sounders", rel="FOLLOWS")

# Teams play at venues
G.add_edge("lafc",     "bmo",   rel="PLAYS_AT")
G.add_edge("lagalaxy", "sofi",  rel="PLAYS_AT")
G.add_edge("sounders", "lumen", rel="PLAYS_AT")

# Events hosted at venues
G.add_edge("el_trafico", "sofi",  rel="HOSTED_AT")
G.add_edge("cascadia",   "lumen", rel="HOSTED_AT")

# Fans attend events
G.add_edge("sarah", "el_trafico", rel="ATTENDS")
G.add_edge("mike",  "el_trafico", rel="ATTENDS")
G.add_edge("priya", "cascadia",   rel="ATTENDS")

# ---------------------------------------------------------
# 2. BASIC QUERIES — traversing the graph
# ---------------------------------------------------------

def neighbors_by_relation(node, relation, direction="out"):
    """Find all neighbors of `node` connected by `relation`."""
    result = []
    if direction == "out":
        edges = G.out_edges(node, data=True)
        result = [v for u, v, d in edges if d.get("rel") == relation]
    else:
        edges = G.in_edges(node, data=True)
        result = [u for u, v, d in edges if d.get("rel") == relation]
    return result


def fans_in_city_who_follow_teams_playing_at(city_id, venue_id):
    """
    Multi-hop query:
      City <-LIVES_IN- Fan -FOLLOWS-> Team -PLAYS_AT-> Venue
    Returns fans matching this chain.
    """
    matches = []
    # Step 1: fans in the city
    fans_in_city = neighbors_by_relation(city_id, "LIVES_IN", direction="in")

    for fan in fans_in_city:
        # Step 2: teams the fan follows
        teams_followed = neighbors_by_relation(fan, "FOLLOWS")
        for team in teams_followed:
            # Step 3: venues where that team plays
            team_venues = neighbors_by_relation(team, "PLAYS_AT")
            if venue_id in team_venues:
                matches.append({
                    "fan": G.nodes[fan]["name"],
                    "team": G.nodes[team]["name"],
                    "venue": G.nodes[venue_id]["name"],
                })
    return matches


# ---------------------------------------------------------
# 3. TRY IT OUT
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("QUERY 1: Who does Sarah follow?")
    print("=" * 55)
    for team in neighbors_by_relation("sarah", "FOLLOWS"):
        print(f"  → {G.nodes[team]['name']}")

    print("\n" + "=" * 55)
    print("QUERY 2: Who attends El Tráfico?")
    print("=" * 55)
    for fan in neighbors_by_relation("el_trafico", "ATTENDS", direction="in"):
        print(f"  → {G.nodes[fan]['name']}")

    print("\n" + "=" * 55)
    print("QUERY 3: Fans in LA who follow teams playing at SoFi")
    print("=" * 55)
    results = fans_in_city_who_follow_teams_playing_at("la", "sofi")
    for r in results:
        print(f"  → {r['fan']} follows {r['team']} ({r['venue']})")

    print("\n" + "=" * 55)
    print("QUERY 4: Shortest path between Sarah and Priya")
    print("=" * 55)
    try:
        # Convert to undirected for path finding
        UG = G.to_undirected()
        path = nx.shortest_path(UG, "sarah", "priya")
        names = [G.nodes[n]["name"] for n in path]
        print("  " + " → ".join(names))
    except nx.NetworkXNoPath:
        print("  No path found")

    # ---------------------------------------------------
    # 4. VISUALIZE
    # ---------------------------------------------------
    print("\nGenerating graph visualization...")
    pos = nx.spring_layout(G, seed=42, k=1.2)

    # Color nodes by type
    color_map = {
        "Fan":   "#4C9AFF",
        "Team":  "#FF6B6B",
        "City":  "#51CF66",
        "Venue": "#FFD43B",
        "Event": "#CC5DE8",
    }
    node_colors = [color_map[G.nodes[n]["type"]] for n in G.nodes()]

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2200, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, alpha=0.6)
    labels = {n: G.nodes[n]["name"] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)

    # Draw edge labels (relation names)
    edge_labels = {(u, v): d["rel"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6)

    plt.title("MLS Fan Knowledge Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("knowledge_graph.png", dpi=150)
    print("Saved to knowledge_graph.png")



#### How to Generalize this Project into Production
# This toy project	              The production version
# networkx.MultiDiGraph	           Neo4j, Amazon Neptune, or TigerGraph
# Hardcoded nodes	                   Entity resolution pipelines merging "Sarah" across ticketing, app, CRM
# neighbors_by_relation()	           Cypher queries: MATCH (f:Fan)-[:LIVES_IN]->(c:City {id:'la'})-...
# 5 node types	                   Dozens: Fan, Segment, Campaign, Seat, Concession, Match, Sponsor...
# Manual edge creation	           Automated enrichment from streaming events
# matplotlib viz	                   Graph exploration UI for product teams