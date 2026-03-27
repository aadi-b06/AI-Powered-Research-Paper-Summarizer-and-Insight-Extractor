"""
gemini_file.py
--------------
Wrapper for Gemini 2.5 Flash API calls.
Used by the Streamlit dashboard for RAG-based Q&A.
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def ask_gemini(content, query):
    """
    Sends retrieved paper content + user query to Gemini.
    Returns a structured Answer + Research Paper citation response.
    """
    prompt = f"""
You are a research assistant.

Use ONLY the provided research paper content to answer the question.

Rules:
1. If the answer exists in the provided papers, generate the answer.
2. Mention ONLY the title of the research paper used for the answer.
3. If the answer is not present in the papers, respond exactly:

Answer: Not found in the retrieved papers.
Research Paper: None

Response format:

Answer:
<answer>

Research Paper:
<paper title>, <paper title> if multiple papers are used

Content:
{content}

Question:
{query}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.9
        ),
    )

    return response.text