"""
extract_pdf.py
--------------
Document Ingestion & Parsing Layer.
Reads PDFs from the 'data/' folder, extracts metadata + full text,
generates summaries and structured insights, and saves JSON to 'parsed_output/'.
"""

import fitz  # PyMuPDF
import re
import json
import uuid
import os
from datetime import datetime
from helper_function import insigth_extraction, summeriser


# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_pdf_text(pdf_path):
    """Ingests PDFs and parses them into raw text."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (b[1], b[0]))  # Sort top-to-bottom, left-to-right
        for b in blocks:
            if b[6] == 0:  # text block (not image)
                text += b[4] + "\n"
    return text


# =========================
# TEXT CLEANING
# =========================
def clean_text(text):
    """Cleans whitespace and fixes hyphenated line breaks."""
    text = text.replace("\r", "\n")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =========================
# TITLE EXTRACTION
# =========================
def extract_title(text):
    """Extracts the research paper title from the top of the document."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    title_lines = []

    for i, line in enumerate(lines[:15]):
        if re.search(r'\bAbstract\b', line, re.IGNORECASE):
            break
        if title_lines and title_lines[-1].endswith(":"):
            title_lines.append(line)
            break
        if re.search(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', line):
            if title_lines:
                break
        if len(line) > 10:
            title_lines.append(line)
        if len(title_lines) == 1 and not line.endswith(":"):
            break

    return " ".join(title_lines) if title_lines else "Unknown Title"


# =========================
# AUTHOR EXTRACTION
# =========================
def extract_authors(text):
    """Identifies proper names while filtering out institutions and emails."""
    abstract_match = re.search(r'\bAbstract\b', text, re.IGNORECASE)
    if not abstract_match:
        return []

    header_text = text[:abstract_match.start()]
    lines = [line.strip() for line in header_text.split("\n") if line.strip()]

    if len(lines) < 2:
        return []

    lines = lines[2:] if len(lines) > 2 else lines[1:]

    cleaned_lines = []
    keywords = ["university", "institute", "department", "correspondence", "preprint", "@", "school", "laborator"]
    for line in lines:
        if any(key in line.lower() for key in keywords):
            continue
        cleaned_lines.append(line)

    author_block = " ".join(cleaned_lines)
    author_block = re.sub(r'[\*\d†‡]', "", author_block)

    authors = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z\-]+){1,2}\b', author_block)

    seen = set()
    final_authors = []
    for name in authors:
        if name not in seen:
            seen.add(name)
            final_authors.append(name)
    return final_authors


# =========================
# ABSTRACT EXTRACTION
# =========================
def extract_abstract(text):
    """Extracts the abstract, stopping at Introduction or the next section."""
    text = text.replace('\r', '\n')
    pattern = re.search(
        r'\bAbstract\b[\s\.\:]*\n?(.*?)\n\s*(?:\d+\.?\s*)?Introduction',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if pattern:
        return clean_text(pattern.group(1).strip())

    fallback = re.search(r'\bAbstract\b[\s\.\:]*\n?(.*)', text, re.IGNORECASE | re.DOTALL)
    if fallback:
        fallback_text = fallback.group(1)
        cutoff = re.search(r'\n\s*(?:\d+\.?\s*)?[A-Z][A-Z\s]{3,}\b', fallback_text)
        if cutoff:
            fallback_text = fallback_text[:cutoff.start()]
        else:
            fallback_text = fallback_text[:2000]
        return clean_text(fallback_text.strip())

    return "Abstract Not Found"


# =========================
# CONTENT EXTRACTION
# =========================
def extract_content(text, abstract):
    """Extracts body content after the abstract, stopping before References."""
    if abstract and abstract in text:
        content = text.split(abstract, 1)[-1]
    else:
        content = text

    ref_match = re.search(r'\bReferences\b', content, re.IGNORECASE)
    if ref_match:
        content = content[:ref_match.start()]

    content = re.sub(r'\s+', ' ', content)
    return content.strip()


# =========================
# CREATE JSON STRUCTURE
# =========================
def create_json_structure(pdf_path, raw_text):
    """Builds the full structured JSON for a single paper."""
    cleaned_text = clean_text(raw_text)
    title = extract_title(raw_text)
    authors = extract_authors(raw_text)
    abstract = extract_abstract(raw_text)
    content = extract_content(cleaned_text, abstract)

    document_id = str(uuid.uuid4())

    return {
        "document_id": document_id,
        "source_file": os.path.basename(pdf_path),
        "metadata": {
            "title": title,
            "authors": authors,
            "publication_year": None,
            "doi": None,
            "keywords": [],
            "created_at": datetime.utcnow().isoformat()
        },
        "abstract": abstract,
        "content": content
    }


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    data_folder = "data"
    output_dir = "parsed_output"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_folder):
        print(f"❌ 'data/' folder not found. Please place your PDF files there.")
        exit()

    # Load BART summarization model
    print("Loading summarization model (facebook/bart-large-cnn)...")
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print("⚠️ No PDF files found in 'data/' folder.")
        exit()

    for file_name in pdf_files:
        pdf_path = os.path.join(data_folder, file_name)
        print(f"\n📄 Processing: {file_name}")

        raw_text = extract_pdf_text(pdf_path)
        paper_data = create_json_structure(pdf_path, raw_text)

        # Summarize
        print("  → Generating summary...")
        paper_data["summary"] = summeriser(paper_data["content"], tokenizer, model)

        # Extract insights
        print("  → Extracting insights...")
        paper_data["insigth"] = insigth_extraction(paper_data["summary"])

        output_path = os.path.join(output_dir, f"{paper_data['document_id']}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(paper_data, f, indent=4)

        print(f"  ✅ Saved: {output_path}")

    print("\n🎉 All PDFs processed successfully!")