import feedparser
import requests
import time
import pandas as pd

def fetch_arxiv_papers(query, max_results=50):
    # Base URL for the arXiv API
    url = "http://export.arxiv.org/api/query"
    
    # Parameters to request exactly what you need
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results
    }
    
    # Send the GET request (The 'Waiter' taking your request)
    response = requests.get(url, params=params)
    
    # Parse the response (The 'Waiter' bringing the food)
    feed = feedparser.parse(response.text)
    
    papers = []
    for entry in feed.entries:
        # Converting raw data into a clean, standardized format
        papers.append({
            "source": "arxiv",
            "paper_id": entry.id,
            "title": entry.title.strip(),
            "authors": ", ".join([a.name for a in entry.authors]),
            "abstract": entry.summary.strip(),
            "published": entry.published,
            "pdf_url": next(
                (link.href for link in entry.links if link.type == "application/pdf"),
                None
            )
        })
    
    time.sleep(3) # Respect rate limits
    return papers

if __name__ == "__main__":
    # Choose a query like 'all:"machine learning"'
    query_topic = 'all:"machine learning"'
    
    print(f"Fetching papers for {query_topic}...")
    papers_list = fetch_arxiv_papers(query_topic, max_results=10)
    
    # Convert the list to a DataFrame (Standardized table)
    df = pd.DataFrame(papers_list)
    
    # Save to Excel
    filename = "arxiv_papers.xlsx"
    df.to_excel(filename, index=False)
    print(f"Saved successfully to {filename}")