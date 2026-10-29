####
# Purpose: To extend knowledge_graph.py to include capacity for achieving 
# retrieval-augmented generation grounded in semantic data

# ### User question (natural language)
#         │
#         ▼
# ┌───────────────────┐
# │ 1. GRAPH LOOKUP   │  → structured facts (who follows whom)
# └───────────────────┘
#         │
#         ▼
# ┌───────────────────┐
# │ 2. VECTOR SEARCH  │  → semantically similar text (docs, notes)
# └───────────────────┘
#         │
#         ▼
# ┌───────────────────┐
# │ 3. CONTEXT ASSEMBLY│ → combine graph facts + retrieved text
# └───────────────────┘
#         │
#         ▼
# ┌───────────────────┐
# │ 4. LLM GENERATION │ → grounded answer with citations
# └───────────────────┘

# ##Important:
# In this project, we use a mock LLM so the project runs 
# offline with no API key. The structure is identical to a 
# real one. For example, to include live LLM, we can swap in 
# openai, anthropic, or a local llama-cpp call and it works the 
# same

import networkx as nx
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =========================================================
# PART 1: THE KNOWLEDGE GRAPH (from before, trimmed)
# =========================================================
G = nx.MultiDiGraph()

nodes = {
    "sarah":     {"type": "Fan",   "name": "Sarah"},
    "mike":      {"type": "Fan",   "name": "Mike"},
    "priya":     {"type": "Fan",   "name": "Priya"},
    "lafc":      {"type": "Team",  "name": "LAFC"},
    "lagalaxy":  {"type": "Team",  "name": "LA Galaxy"},
    "sounders":  {"type": "Team",  "name": "Seattle Sounders"},
    "la":        {"type": "City",  "name": "Los Angeles"},
    "sea":       {"type": "City",  "name": "Seattle"},
    "bmo":       {"type": "Venue", "name": "BMO Stadium"},
    "sofi":      {"type": "Venue", "name": "SoFi Stadium"},
    "lumen":     {"type": "Venue", "name": "Lumen Field"},
}
for nid, attrs in nodes.items():
    G.add_node(nid, **attrs)

edges = [
    ("sarah", "la",        "LIVES_IN"),
    ("mike",  "la",        "LIVES_IN"),
    ("priya", "sea",       "LIVES_IN"),
    ("sarah", "lafc",      "FOLLOWS"),
    ("mike",  "lagalaxy",  "FOLLOWS"),
    ("priya", "sounders",  "FOLLOWS"),
    ("lafc",     "bmo",    "PLAYS_AT"),
    ("lagalaxy", "sofi",   "PLAYS_AT"),
    ("sounders", "lumen",  "PLAYS_AT"),
]
for u, v, rel in edges:
    G.add_edge(u, v, rel=rel)


def describe_node(node_id):
    """Turn a node + its edges into a natural-language fact sentence."""
    attrs = G.nodes[node_id]
    facts = []
    for _, target, data in G.out_edges(node_id, data=True):
        target_name = G.nodes[target]["name"]
        rel = data["rel"]
        facts.append(f"{attrs['name']} {rel.lower().replace('_', ' ')} {target_name}.")
    return f"{attrs['name']} is a {attrs['type']}. " + " ".join(facts)


# =========================================================
# PART 2: BUILD THE VECTOR INDEX
# =========================================================
print("Loading embedding model...")
# Small, fast, runs on CPU. In production you'd use a bigger model
# or an API (OpenAI text-embedding-3, Cohere, etc.)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- The corpus: every graph node described in English ---
# PLUS extra unstructured docs (notes, policies) that live outside the graph.
documents = []
doc_ids = []

for node_id in G.nodes():
    documents.append(describe_node(node_id))
    doc_ids.append(("node", node_id))

# Unstructured docs — the kind of thing a semantic layer needs to unify
extra_docs = [
    ("policy", "flex_pricing",
     "Flex pricing: LAFC dynamically adjusts ticket prices based on demand. "
     "Fans in Los Angeles who follow LAFC may see premium pricing for rivalry matches."),
    ("policy", "member_perks",
     "Season members get 15% off concessions at BMO Stadium and priority access to playoff tickets."),
    ("note",   "mike_note",
     "Account note for Mike: called support about parking at SoFi Stadium. "
     "Mentioned he's a long-time LA Galaxy fan and attends most home games."),
    ("note",   "sarah_note",
     "Account note for Sarah: opted into LAFC email but not SMS. "
     "Interested in family-friendly seating and Black-and-Gold merchandise."),
]
for kind, doc_id, text in extra_docs:
    documents.append(text)
    doc_ids.append((kind, doc_id))

print(f"Embedding {len(documents)} documents...")
embeddings = embedder.encode(documents, convert_to_numpy=True, normalize_embeddings=True)

# FAISS index — inner product on normalized vectors == cosine similarity
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype("float32"))

# =========================================================
# PART 3: GRAPH-AWARE RETRIEVAL
# =========================================================
def graph_lookup(query: str):
    """
    Toy 'entity linker': find nodes whose name appears in the query.
    Production systems use NER + entity resolution here.
    """
    query_lower = query.lower()
    matched = []
    for nid, attrs in G.nodes(data=True):
        if attrs["name"].lower() in query_lower:
            matched.append(nid)
    return matched


def graph_context(node_ids, hops=1):
    """
    Expand matched nodes by N hops and return their fact sentences.
    This is the 'context assembly' the job description mentions.
    """
    # BFS expansion
    visited = set(node_ids)
    frontier = set(node_ids)
    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            for _, tgt, _ in G.out_edges(n, data=True):
                if tgt not in visited:
                    next_frontier.add(tgt)
            for src, _, _ in G.in_edges(n, data=True):
                if src not in visited:
                    next_frontier.add(src)
        visited |= next_frontier
        frontier = next_frontier

    facts = [describe_node(n) for n in visited]
    return visited, facts


def vector_search(query: str, k: int = 3):
    """Return top-k documents by cosine similarity."""
    q_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = index.search(q_vec.astype("float32"), k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            "score": float(score),
            "doc": documents[idx],
            "source": doc_ids[idx],
        })
    return results


# =========================================================
# PART 4: MOCK LLM
# =========================================================
def mock_llm(prompt: str) -> str:
    """
    Placeholder for a real LLM call. Notice the prompt already contains
    the retrieved context — this is the 'augmented' half of RAG.

    To swap in a real model:
        # OpenAI
        from openai import OpenAI
        client = OpenAI()
        return client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

        # Or a local open-weight model via llama-cpp-python
        # (the "on-prem GPU" scenario in the job description)
    """
    print("\n----- PROMPT SENT TO LLM -----")
    print(prompt[:1200] + ("\n... (truncated)" if len(prompt) > 1200 else ""))
    print("------------------------------\n")
    return "[Mock LLM] A real model would answer here, grounded in the context above."


# =========================================================
# PART 5: THE RAG PIPELINE: Graph + Vector + RAG
# =========================================================
def ask(question: str):
    print("\n" + "=" * 60)
    print(f"Q: {question}")
    print("=" * 60)

    # 1. Graph lookup + expansion
    entities = graph_lookup(question)
    expanded, graph_facts = graph_context(entities, hops=1)

    # 2. Vector search
    retrieved = vector_search(question, k=3)

    # 3. Assemble context
    context_blocks = []
    if graph_facts:
        context_blocks.append("GRAPH FACTS:\n" + "\n".join(f"- {f}" for f in graph_facts))
    context_blocks.append(
        "RETRIEVED DOCUMENTS:\n"
        + "\n".join(f"- [{r['source'][0]}:{r['source'][1]}] {r['doc']}" for r in retrieved)
    )
    context = "\n\n".join(context_blocks)

    prompt = f"""You are an MLS fan intelligence assistant.
Answer the user's question using ONLY the context below.
If the answer isn't in the context, say so.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    answer = mock_llm(prompt)

    # 4. Show what grounded the answer — traceability!
    print("Graph entities matched:", [G.nodes[e]["name"] for e in entities] or "none")
    print("Graph expansion:", [G.nodes[e]["name"] for e in expanded])
    print("\nTop retrieved docs:")
    for r in retrieved:
        print(f"  [{r['score']:.3f}] ({r['source'][0]}:{r['source'][1]}) {r['doc'][:90]}...")
    print(f"\n{answer}")
    return {"entities": entities, "expanded": expanded, "retrieved": retrieved, "answer": answer}


# =========================================================
# PART 6: DEMO QUERIES
# =========================================================
if __name__ == "__main__":
    ask("What team does Sarah follow and where does she live?")
    ask("Tell me about Mike's support history and any venue issues.")
    ask("What perks does Sarah get as a season member at BMO Stadium?")
    ask("Which fans follow teams in Seattle?")