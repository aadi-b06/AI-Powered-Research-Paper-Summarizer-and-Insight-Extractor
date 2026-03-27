import fitz
import re

def extract_pdf_text(pdf_path):
    """Ingests PDFs and parses them into raw text, handling two-column layouts."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        # Extract text in blocks to maintain logical reading order
        blocks = page.get_text("blocks")
        # Sort blocks vertically (y0), then horizontally (x0)
        blocks.sort(key=lambda b: (b[1], b[0]))
        for b in blocks:
            if b[6] == 0:  # block type 0 means it is text, not an image
                text += b[4] + "\n"
    return text

def clean_text(text):
    """Cleans whitespace and fixes hyphenated line breaks."""
    # Fix hyphenated words at the end of lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_title(text):
    """Extracts the research paper title from the top of the document."""
    lines = text.split("\n")
    for line in lines[:20]:
        cleaned_line = line.strip()
        # Skip common academic header artifacts
        if "arxiv" in cleaned_line.lower() or "journal" in cleaned_line.lower():
            continue
        # Assume the first reasonably long string without a year is the title
        if len(cleaned_line) > 10 and not re.search(r'\d{4}', cleaned_line):
            return cleaned_line
    return "Unknown title"

def extract_authors(text):
    """Identifies proper names while filtering out institutions and emails."""
    abstract_match = re.search(r'\bAbstract\b', text, re.IGNORECASE)
    if not abstract_match:
        return []
    
    header_text = text[:abstract_match.start()]
    lines = [line.strip() for line in header_text.split("\n") if line.strip()]
    if len(lines) < 2: 
        return []

    # Remove title line and filter keywords
    lines = lines[1:]
    cleaned_lines = []
    keywords = ["university", "institute", "department", "correspondence", "preprint", "@", "school", "laborator"]
    for line in lines:
        if any(key in line.lower() for key in keywords):
            continue
        cleaned_lines.append(line)
    
    author_block = " ".join(cleaned_lines)
    author_block = re.sub(r'[\*\d†‡]', "", author_block) # Remove footnote symbols/numbers
    
    # Use Regex to find names (First Last, or First M. Last)
    names = re.findall(r'\b[A-Z][a-zA-Z\-\.]+(?:\s[A-Z][a-zA-Z\-\.]+){1,3}\b', author_block)
    
    # Remove duplicates while preserving order
    final_authors = []
    seen = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            final_authors.append(name)
    return final_authors

def extract_abstract(text):
    """Extracts the abstract, stopping at Introduction or the next section."""
    text = text.replace('\r', '\n')
    
    # Look for 'Abstract' and end at 'Introduction'
    pattern = re.search(
        r'\bAbstract\b[\s\.\:]*\n?(.*?)\n\s*(?:\d+\.?\s*)?Introduction', 
        text, 
        re.IGNORECASE | re.DOTALL
    )

    if pattern:
        abstract = pattern.group(1).strip()
        return clean_text(abstract)
    
    # Fallback: If 'Introduction' isn't found
    fallback_match = re.search(r'\bAbstract\b[\s\.\:]*\n?(.*)', text, re.IGNORECASE | re.DOTALL)
    if fallback_match:
        fallback_text = fallback_match.group(1)
        # Try to cut off at the first major section header (usually all caps or numbered)
        cutoff = re.search(r'\n\s*(?:\d+\.?\s*)?[A-Z][A-Z\s]{3,}\b', fallback_text)
        if cutoff:
            fallback_text = fallback_text[:cutoff.start()]
        else:
            fallback_text = fallback_text[:2000]
        return clean_text(fallback_text.strip())
        
    return "Abstract not Found"