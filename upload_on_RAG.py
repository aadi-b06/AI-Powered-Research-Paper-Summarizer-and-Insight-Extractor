"""
upload_on_RAG.py
----------------
RAG + Semantic Search Layer.
Loads all paper data (parsed PDFs + arXiv + PubMed),
builds a FAISS vector store, and saves it locally.
Run this ONCE before launching the Streamlit dashboard.
"""

import json
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


documents = []
metadatas = []

# ─────────────────────────────────────────────
# STEP 1: Load data from parsed PDF output folder
# ─────────────────────────────────────────────
parsed_folder = "parsed_output"

if os.path.exists(parsed_folder):
    for filename in os.listdir(parsed_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(parsed_folder, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                metadata = data.get("metadata", {})
                insight = data.get("insigth", {})  # Note: typo preserved for compatibility

                text = f"""
Title: {metadata.get('title', '')}
Authors: {", ".join(metadata.get('authors', []))}
Publication Year: {metadata.get('publication_year', '')}
DOI: {metadata.get('doi', '')}
Keywords: {", ".join(metadata.get('keywords', []))}

Domain: {", ".join(insight.get("domain", [])) if insight else ""}
Research Problem: {insight.get("research_problem", "") if insight else ""}
Methods: {", ".join(insight.get("methods", [])) if insight else ""}
Datasets: {", ".join(insight.get("datasets", [])) if insight else ""}
Metrics: {", ".join(insight.get("metrics", [])) if insight else ""}
Key Findings: {insight.get("key_findings", "") if insight else ""}
Limitations: {insight.get("limitations", "") if insight else ""}
Future Directions: {insight.get("future_directions", "") if insight else ""}

Abstract:
{data.get("abstract", "")}

Summary:
{data.get("summary", "")}
""".strip()

                documents.append(text)
                metadatas.append({
                    "paper_id": data.get("document_id"),
                    "title": metadata.get("title"),
                    "source": data.get("source_file"),
                    "publication_year": metadata.get("publication_year"),
                    "domain": insight.get("domain", []) if insight else [],
                })

            except Exception as e:
                print(f"⚠️ Error processing file {filename}: {e}")

    print(f"✅ Loaded {len(documents)} documents from parsed_output/")
else:
    print("⚠️ 'parsed_output/' folder not found. Skipping PDF papers.")


# ─────────────────────────────────────────────
# STEP 2: Load arXiv papers
# ─────────────────────────────────────────────
arxiv_file = "arxiv_papers.json"
if os.path.exists(arxiv_file):
    with open(arxiv_file, "r", encoding="utf-8") as f:
        arxiv_papers = json.load(f)

    for paper in arxiv_papers:
        insight = paper.get("insight", {}) or {}

        text = f"""
Title: {paper.get('title', '')}
Authors: {paper.get('authors', '')}
Published: {paper.get('published', '')}
Categories: {paper.get('categories', '')}

Domain: {", ".join(insight.get("domain", []))}
Research Problem: {insight.get("research_problem", "")}
Methods: {", ".join(insight.get("methods", []))}
Datasets: {", ".join(insight.get("datasets", []))}
Metrics: {", ".join(insight.get("metrics", []))}
Key Findings: {insight.get("key_findings", "")}
Limitations: {insight.get("limitations", "")}
Future Directions: {insight.get("future_directions", "")}

Abstract:
{paper.get("abstract", "")}
""".strip()

        documents.append(text)
        metadatas.append({
            "paper_id": paper.get("paper_id"),
            "title": paper.get("title"),
            "source": paper.get("source"),
            "categories": paper.get("categories"),
            "domain": insight.get("domain", []),
        })

    print(f"✅ Loaded {len(arxiv_papers)} arXiv papers")
else:
    print("⚠️ 'arxiv_papers.json' not found. Run 'data_ingest.py' first.")


# ─────────────────────────────────────────────
# STEP 3: Load PubMed papers
# ─────────────────────────────────────────────
pubmed_file = "pubmed_multiple_queries.json"
if os.path.exists(pubmed_file):
    with open(pubmed_file, "r", encoding="utf-8") as f:
        pubmed_papers = json.load(f)

    for paper in pubmed_papers:
        insight = paper.get("insight", {}) or {}

        text = f"""
Title: {paper.get('title', '')}
Authors: {", ".join(paper.get('authors', [])) if isinstance(paper.get('authors'), list) else paper.get('authors', '')}
Journal: {paper.get('journal', '')}
Keywords: {", ".join(paper.get('keywords', []))}

Domain: {", ".join(insight.get("domain", []))}
Research Problem: {insight.get("research_problem", "")}
Methods: {", ".join(insight.get("methods", []))}
Datasets: {", ".join(insight.get("datasets", []))}
Metrics: {", ".join(insight.get("metrics", []))}
Key Findings: {insight.get("key_findings", "")}
Limitations: {insight.get("limitations", "")}
Future Directions: {insight.get("future_directions", "")}

Abstract:
{paper.get("abstract", "")}
""".strip()

        documents.append(text)
        metadatas.append({
            "paper_id": paper.get("pmid"),
            "title": paper.get("title"),
            "source": "pubmed",
            "journal": paper.get("journal", ""),
            "domain": insight.get("domain", []),
        })

    print(f"✅ Loaded {len(pubmed_papers)} PubMed papers")
else:
    print("⚠️ 'pubmed_multiple_queries.json' not found. Run 'pubmed.py' first.")


# ─────────────────────────────────────────────
# STEP 4: Build and save FAISS vector store
# ─────────────────────────────────────────────
print(f"\n📦 Total documents to embed: {len(documents)}")

if len(documents) == 0:
    print("❌ No documents found. Please run extract_pdf.py, data_ingest.py, and pubmed.py first.")
    exit()

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Building FAISS index...")
vector_db = FAISS.from_texts(
    texts=documents,
    embedding=embeddings,
    metadatas=metadatas
)

print(f"✅ Vectors in index: {vector_db.index.ntotal}")

vector_db.save_local("research_papers_faiss")
print("💾 FAISS index saved to 'research_papers_faiss/' successfully!")