import torch
import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def summeriser(text, tokenizer, model):
    """Summarizes text using a HuggingFace Seq2Seq model (e.g. BART)."""
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=1000,
            min_length=300,
            length_penalty=0.8,
            num_beams=5,
            early_stopping=True
        )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("summary: ", summary)
    return summary


def insigth_extraction(summary):
    """Extracts structured insights from an abstract/summary using Groq LLM."""
    if not summary:
        return {}

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""
    Extract structured insights from the abstract below.

    Return ONLY valid JSON.
    Do not add explanations.
    Do not add markdown.
    Do not add text before or after JSON.

    Use this exact format:

    {{
    "domain": [],
    "research_problem": "",
    "methods": [],
    "datasets": [],
    "metrics": [],
    "key_findings": "",
    "limitations": "",
    "future_directions": ""
    }}

    Abstract:
    {summary}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response.choices[0].message.content
        # Strip markdown code fences if present
        content = content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        return json.loads(content)
    except Exception as e:
        print(f"⚠️ Insight extraction error: {e}")
        return {}