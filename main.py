"""
main.py
-------
Main Streamlit Dashboard.
Run with: streamlit run main.py

Tabs:
  1. Research Paper QA   — RAG-based Q&A using FAISS + Gemini
  2. Knowledge Graph Explorer — Neo4j graph visualization by domain

Requirements:
  - research_papers_faiss/  (run upload_on_RAG.py first)
  - Neo4j running on localhost:7687  (run upload_on_neo4j.py first)
"""

import streamlit as st
import io
import pandas as pd

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from gemini_file import ask_gemini

from neo4j import GraphDatabase
from pyvis.network import Network
import streamlit.components.v1 as components

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Intelligence Platform",
    page_icon="🔬",
    layout="wide"
)

# Title
st.markdown(
    "<h1 style='text-align:center;'>🔬 AI-Powered Research Paper Summarizer & Insight Extractor</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:gray;'>Ask questions and get AI-powered insights from research papers</p>",
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["📄 Research Paper QA", "🕸️ Knowledge Graph Explorer"])


# ═══════════════════════════════════════════════
# TAB 1: RAG Research Paper Q&A
# ═══════════════════════════════════════════════
with tab1:

    st.markdown("""
    <style>
    .stTextInput input {
        background-color: #F5F5F5;
        color: #000000;
        border-radius: 10px;
        border: 2px solid #4A90D9;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    @st.cache_resource
    def load_vector_db():
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_db = FAISS.load_local(
            "research_papers_faiss",
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_db

    try:
        vector_db = load_vector_db()
        st.success(f"📚 Vector database loaded — {vector_db.index.ntotal} paper chunks indexed")
    except Exception as e:
        st.error(f"❌ Could not load vector database: {e}")
        st.info("Run `python upload_on_RAG.py` first to build the database.")
        st.stop()

    user_query = st.text_input("🔎 Ask a question about research papers:")

    if st.button("🚀 Search & Generate Insight", key="search_btn"):
        if not user_query.strip():
            st.warning("Please enter a question first.")
        else:
            results = vector_db.similarity_search(user_query, k=3)

            content = ""
            for idx, doc in enumerate(results, 1):
                title = doc.metadata.get("title", f"Paper {idx}")
                content += f"""
Paper Title: {title}

Paper Content:
{doc.page_content}

"""

            with st.spinner("🤖 Analyzing research papers and generating insights..."):
                response = ask_gemini(content, user_query)

            # Parse Answer and Paper Titles
            answer = ""
            paper_titles = []

            if "Research Paper:" in response:
                parts = response.split("Research Paper:")
                answer = parts[0].replace("Answer:", "").strip()
                papers_text = parts[1].strip()
                paper_titles = [p.strip() for p in papers_text.split(",")]
            else:
                answer = response.strip()

            # Display AI answer
            st.subheader("🤖 AI Generated Insight")
            st.write(answer)

            # Display relevant paper snippets
            if paper_titles and "none" not in [p.lower() for p in paper_titles]:
                st.subheader("📄 Relevant Research Papers")
                for doc in results:
                    title = doc.metadata.get("title", "")
                    for p in paper_titles:
                        if title and title.lower() == p.lower():
                            with st.expander(f"📄 {title}"):
                                st.write(doc.page_content)
            else:
                st.info("No exact paper match found in top results — answer generated from closest context.")


# ═══════════════════════════════════════════════
# TAB 2: Knowledge Graph Explorer
# ═══════════════════════════════════════════════
with tab2:
    st.subheader("🕸️ Knowledge Graph Explorer")
    st.write("Explore relationships between research papers, authors, methods, and domains.")

    # Neo4j connection
    @st.cache_resource
    def get_driver():
        try:
            return GraphDatabase.driver(
                "neo4j://127.0.0.1:7687",
                auth=("neo4j", "neo4j123")
            )
        except Exception as e:
            return None

    driver = get_driver()

    if driver is None:
        st.error("❌ Could not connect to Neo4j. Make sure Neo4j Desktop is running on localhost:7687.")
        st.info("Run `python upload_on_neo4j.py` after starting Neo4j to populate the graph.")
    else:

        @st.cache_data
        def get_domains():
            query = "MATCH (d:Domain) RETURN d.name AS domain ORDER BY d.name"
            try:
                with driver.session() as session:
                    result = session.run(query)
                    domains = [r["domain"] for r in result if r["domain"]]
                return sorted(set(d.title() for d in domains))
            except Exception as e:
                return []

        domains = get_domains()

        if not domains:
            st.warning("⚠️ No domains found in the graph. Run `python upload_on_neo4j.py` first.")
        else:
            domain = st.selectbox("🔍 Select Research Domain", domains)

            def get_graph_data(selected_domain):
                query = """
                MATCH (p:Paper)-[:BELONGS_TO]->(d:Domain)
                WHERE toLower(d.name) = toLower($domain)
                OPTIONAL MATCH (p)<-[:WROTE]-(a:Author)
                OPTIONAL MATCH (p)-[:USES]->(m:Method)
                RETURN p.title AS paper,
                       a.name AS author,
                       m.name AS method,
                       d.name AS domain
                """
                with driver.session() as session:
                    result = session.run(query, domain=selected_domain)
                    return [r.data() for r in result]

            def draw_graph(data):
                net = Network(
                    height="600px",
                    width="100%",
                    bgcolor="#1a1a2e",
                    font_color="white"
                )
                net.set_options("""
                {
                  "nodes": {"font": {"size": 12}},
                  "physics": {"stabilization": {"iterations": 100}}
                }
                """)

                added_nodes = set()

                for row in data:
                    paper = row.get("paper")
                    author = row.get("author")
                    method = row.get("method")
                    domain_name = row.get("domain")

                    if paper and paper not in added_nodes:
                        net.add_node(paper, label=paper[:40] + ("..." if len(paper) > 40 else ""),
                                     color="#FF8C00", title=paper, shape="box")
                        added_nodes.add(paper)

                    if author and author not in added_nodes:
                        net.add_node(author, label=author, color="#00BFFF", title=f"Author: {author}")
                        added_nodes.add(author)
                    if author and paper:
                        net.add_edge(author, paper, label="WROTE", color="#00BFFF")

                    if method and method not in added_nodes:
                        net.add_node(method, label=method, color="#00C853", title=f"Method: {method}")
                        added_nodes.add(method)
                    if method and paper:
                        net.add_edge(paper, method, label="USES", color="#00C853")

                    if domain_name and domain_name not in added_nodes:
                        net.add_node(domain_name, label=domain_name, color="#AB47BC",
                                     title=f"Domain: {domain_name}", shape="diamond")
                        added_nodes.add(domain_name)
                    if domain_name and paper:
                        net.add_edge(paper, domain_name, label="BELONGS_TO", color="#AB47BC")

                net.save_graph("graph.html")
                with open("graph.html", "r", encoding="utf-8") as f:
                    components.html(f.read(), height=620)

            if domain:
                st.subheader(f"📊 Knowledge Graph for Domain: **{domain}**")

                data = get_graph_data(domain)

                if not data:
                    st.warning(f"No papers found for domain: {domain}")
                else:
                    df = pd.DataFrame(data)

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("📄 Total Papers", df["paper"].nunique())
                    col2.metric("👤 Total Authors", df["author"].nunique())
                    col3.metric("🔧 Total Methods", df["method"].nunique())

                    st.divider()

                    # Data table
                    st.subheader("📋 Filtered Research Data")
                    st.dataframe(df, use_container_width=True)

                    # Export Excel
                    excel_buffer = io.BytesIO()
                    df.to_excel(excel_buffer, index=False)
                    excel_buffer.seek(0)

                    st.download_button(
                        label="📥 Export to Excel",
                        data=excel_buffer,
                        file_name=f"{domain.replace(' ', '_')}_research_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    st.divider()

                    # Graph visualization
                    st.subheader("🕸️ Knowledge Graph Visualization")
                    with st.spinner("Rendering graph..."):
                        draw_graph(data)