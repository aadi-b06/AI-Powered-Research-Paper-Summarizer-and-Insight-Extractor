"""
upload_on_neo4j.py
------------------
Knowledge Graph Builder.
Loads all paper data (arXiv + PubMed + parsed PDFs),
and creates a Neo4j Knowledge Graph with nodes:
  Paper, Author, Domain, Method, Metric
and relationships:
  (Author)-[:WROTE]->(Paper)
  (Paper)-[:BELONGS_TO]->(Domain)
  (Paper)-[:USES]->(Method)
  (Paper)-[:EVALUATED_BY]->(Metric)

Prerequisites:
  - Neo4j Desktop running on localhost:7687
  - pip install neo4j
"""

import json
import os
from neo4j import GraphDatabase


# ─────────────────────────────────────────────
# 1. Connect to Neo4j
# ─────────────────────────────────────────────
driver = GraphDatabase.driver(
    "neo4j://127.0.0.1:7687",
    auth=("neo4j", "neo4j123")
)


# ─────────────────────────────────────────────
# 2. Load all data
# ─────────────────────────────────────────────
data = []

# arXiv papers
if os.path.exists("arxiv_papers.json"):
    with open("arxiv_papers.json", "r", encoding="utf-8") as f:
        arxiv_data = json.load(f)
    data.extend(arxiv_data)
    print(f"✅ Loaded {len(arxiv_data)} arXiv papers")
else:
    print("⚠️ arxiv_papers.json not found")

# PubMed papers
if os.path.exists("pubmed_multiple_queries.json"):
    with open("pubmed_multiple_queries.json", "r", encoding="utf-8") as f:
        pubmed_data = json.load(f)
    data.extend(pubmed_data)
    print(f"✅ Loaded {len(pubmed_data)} PubMed papers")
else:
    print("⚠️ pubmed_multiple_queries.json not found")

# Parsed PDF JSONs
folder_path = "parsed_output"
if os.path.exists(folder_path):
    pdf_count = 0
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                parsed = json.load(f)
            paper = {
                "title": parsed.get("metadata", {}).get("title"),
                "authors": parsed.get("metadata", {}).get("authors", []),
                "insight": parsed.get("insigth", {})  # typo preserved
            }
            data.append(paper)
            pdf_count += 1
    print(f"✅ Loaded {pdf_count} parsed PDF papers")
else:
    print("⚠️ parsed_output/ not found")

print(f"\n📦 Total papers to insert: {len(data)}")


# ─────────────────────────────────────────────
# 3. Create Graph Function
# ─────────────────────────────────────────────
def create_graph(tx, paper):
    title = paper.get("title")
    if not title:
        return  # Skip papers with no title

    # Create Paper node
    tx.run("MERGE (p:Paper {title: $title})", title=title)

    # Authors
    authors = paper.get("authors", [])
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(",") if a.strip()]

    for author in authors:
        if author.strip():
            tx.run(
                """
                MERGE (a:Author {name: $author})
                MERGE (p:Paper {title: $title})
                MERGE (a)-[:WROTE]->(p)
                """,
                author=author.strip(),
                title=title
            )

    # Insights: Domain, Methods, Metrics
    insight = paper.get("insight", {}) or {}

    for domain in insight.get("domain", []):
        if domain:
            tx.run(
                """
                MERGE (d:Domain {name: $domain})
                MERGE (p:Paper {title: $title})
                MERGE (p)-[:BELONGS_TO]->(d)
                """,
                domain=domain,
                title=title
            )

    for method in insight.get("methods", []):
        if method:
            tx.run(
                """
                MERGE (m:Method {name: $method})
                MERGE (p:Paper {title: $title})
                MERGE (p)-[:USES]->(m)
                """,
                method=method,
                title=title
            )

    for metric in insight.get("metrics", []):
        if metric:
            tx.run(
                """
                MERGE (m:Metric {name: $metric})
                MERGE (p:Paper {title: $title})
                MERGE (p)-[:EVALUATED_BY]->(m)
                """,
                metric=metric,
                title=title
            )


# ─────────────────────────────────────────────
# 4. Insert data into Neo4j
# ─────────────────────────────────────────────
with driver.session() as session:
    for i, paper in enumerate(data):
        session.execute_write(create_graph, paper)
        if (i + 1) % 50 == 0:
            print(f"  Inserted {i + 1}/{len(data)} papers...")

print("\n🎉 Knowledge Graph created successfully in Neo4j!")
print("Open http://localhost:7474 to explore the graph.")