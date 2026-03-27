# 🔬 AI-Powered Research Paper Summarizer & Insight Extractor

An AI-powered research intelligence platform that ingests academic papers, extracts structured insights, enables semantic search via RAG, and visualizes knowledge graphs.

---

## 📋 Project Overview

| Item | Detail |
|------|--------|
| **Goal** | Automate research paper summarization, insight extraction, and knowledge graph construction |
| **Stack** | Python, Streamlit, Gemini 2.5, Groq (Llama), FAISS, Neo4j, HuggingFace BART |
| **Data Sources** | arXiv API, PubMed eUtils API, Local PDFs |

---

## 🗂️ Project Structure

```
├── data/                        # Place your PDF research papers here
├── parsed_output/               # Auto-generated JSON files from PDFs
├── research_papers_faiss/       # Auto-generated FAISS vector store
├── extract_pdf.py               # Module 1: PDF ingestion & parsing
├── data_ingest.py               # Module 1b: arXiv data collection
├── pubmed.py                    # Module 1c: PubMed data collection
├── helper_function.py           # Shared utilities (summarizer + insight extractor)
├── upload_on_RAG.py             # Module 3: Build FAISS vector store
├── upload_on_neo4j.py           # Module 4: Build Neo4j knowledge graph
├── gemini_file.py               # Gemini API wrapper
├── main.py                      # Module 5: Streamlit dashboard (main entry point)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo & install dependencies
```bash
git clone https://github.com/YOUR_USERNAME/research-intelligence-platform.git
cd research-intelligence-platform
pip install -r requirements.txt
```

### 2. Configure environment variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Add PDF papers
Place PDF files inside the `data/` folder.

### 4. Run the data pipeline (in order)

```bash
# Step 1: Parse PDFs → parsed_output/
python extract_pdf.py

# Step 2: Fetch arXiv papers → arxiv_papers.json
python data_ingest.py

# Step 3: Fetch PubMed papers → pubmed_multiple_queries.json
python pubmed.py

# Step 4: Build FAISS vector store
python upload_on_RAG.py

# Step 5: Build Neo4j Knowledge Graph
#   (Start Neo4j Desktop first on localhost:7687)
python upload_on_neo4j.py
```

### 5. Launch the dashboard
```bash
streamlit run main.py
```

---

## 🏗️ Modules

### Module 1: Document Ingestion & Parsing
- `extract_pdf.py` — Reads PDFs, extracts title/authors/abstract/content, generates BART summaries and LLM insights
- `data_ingest.py` — Fetches papers from arXiv API with insight extraction
- `pubmed.py` — Fetches papers from NCBI PubMed with insight extraction

### Module 2: Summarization & Insight Extraction
- `helper_function.py` — BART-based summarizer + Groq LLM structured insight extractor
- Insight schema: `domain`, `research_problem`, `methods`, `datasets`, `metrics`, `key_findings`, `limitations`, `future_directions`

### Module 3: RAG + Semantic Search
- `upload_on_RAG.py` — Embeds all papers using `all-MiniLM-L6-v2` and stores in FAISS
- `gemini_file.py` — Gemini 2.5 Flash for answer generation from retrieved context

### Module 4: Knowledge Graph Builder
- `upload_on_neo4j.py` — Creates Neo4j graph with nodes: `Paper`, `Author`, `Domain`, `Method`, `Metric`
- Relationships: `WROTE`, `BELONGS_TO`, `USES`, `EVALUATED_BY`

### Module 5: Visualization & Dashboard
- `main.py` — Streamlit app with two tabs:
  1. **Research Paper QA** — Semantic search + AI-generated answers
  2. **Knowledge Graph Explorer** — Domain-filtered interactive graph, data table, Excel export

---

## 📅 Agile Development Plan

### Sprint 1 — Data Collection & Parsing
- [x] arXiv API integration
- [x] PubMed API integration
- [x] PDF text extraction (title, authors, abstract, content)
- [x] Structured insight extraction with Groq LLM

### Sprint 2 — RAG Pipeline
- [x] HuggingFace sentence-transformers embeddings
- [x] FAISS vector store creation and persistence
- [x] Semantic similarity search
- [x] Gemini 2.5 Flash integration for RAG Q&A

### Sprint 3 — Knowledge Graph
- [x] Neo4j graph schema design (Paper, Author, Domain, Method, Metric)
- [x] Graph population from arXiv + PubMed + PDF data
- [x] Cypher queries for domain-based traversal

### Sprint 4 — Dashboard & Deployment
- [x] Streamlit dashboard with RAG Q&A tab
- [x] Knowledge Graph Explorer tab with PyVis visualization
- [x] Domain-based filtering and metrics
- [x] Excel export functionality
- [ ] GitHub deployment
- [ ] Final session presentation

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM (Insight Extraction) | Groq — Llama 3.1 8B |
| LLM (RAG Answer) | Google Gemini 2.5 Flash |
| Summarization | Facebook BART-Large-CNN |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| Graph Database | Neo4j |
| Graph Visualization | PyVis |
| Dashboard | Streamlit |
| PDF Parsing | PyMuPDF (fitz) |

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.