"""
data_ingest.py
--------------
Data Ingestion Layer for arXiv papers.
Fetches papers via the arXiv API, extracts structured insights,
and saves results to 'arxiv_papers.json'.
"""

import requests
import feedparser
import json
import time
from helper_function import insigth_extraction


def fetch_arxiv_papers(query, max_results=20, start=0):
    """Fetch papers from arXiv API for a given query."""
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results
    }

    response = requests.get(url, params=params)
    feed = feedparser.parse(response.text)

    papers = []
    for entry in feed.entries:
        abstract = entry.summary.strip()
        print(f"  → Extracting insights for: {entry.title.strip()[:60]}...")
        papers.append({
            "source": "arxiv",
            "search_query": query,
            "paper_id": entry.id,
            "title": entry.title.strip(),
            "authors": ", ".join([a.name for a in entry.authors]),
            "abstract": abstract,
            "published": entry.published,
            "categories": ", ".join([tag.term for tag in entry.tags]),
            "pdf_url": next(
                (link.href for link in entry.links if link.type == "application/pdf"),
                None
            ),
            "insight": insigth_extraction(abstract)
        })

    time.sleep(3)  # Respect arXiv rate limit
    return papers


# ----------------------------
# SEARCH QUERIES
# ----------------------------
queries = [
    'all:"machine learning"',
    'all:"large language model"',
    'all:"generative AI"',
    'all:"retrieval augmented generation" OR all:RAG',
    'all:"semantic search"',
    'all:"vector database"',
    'all:"sentence transformer"'
]

if __name__ == "__main__":
    all_papers = []

    for query in queries:
        print(f"\n🔎 Fetching papers for: {query}")
        papers = fetch_arxiv_papers(query, max_results=20)
        all_papers.extend(papers)
        print(f"  ✅ Fetched {len(papers)} papers")

    # Remove duplicates by paper_id
    unique_papers = {paper["paper_id"]: paper for paper in all_papers}
    final_papers = list(unique_papers.values())

    with open("arxiv_papers.json", "w", encoding="utf-8") as f:
        json.dump(final_papers, f, indent=4, ensure_ascii=False)

    print(f"\n💾 Saved {len(final_papers)} unique papers to arxiv_papers.json ✅")