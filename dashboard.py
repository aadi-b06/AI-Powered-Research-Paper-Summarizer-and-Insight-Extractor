import streamlit as st
import os
import json 
from dotenv import load_dotenv
from groq import Groq
from Data_extract_from_PDF import * 
load_dotenv() 
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="AI Research Assistant", layout="wide")
st.title("📄 Research Paper Insight Extractor")

uploaded_file = st.file_uploader("Upload a Research PDF", type="pdf")

if uploaded_file:   
    with st.spinner("Extracting metadata..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        raw_text = extract_pdf_text("temp.pdf")
        title = extract_title(raw_text)
        authors = extract_authors(raw_text)
        abstract = extract_abstract(raw_text)

    st.header(f"Title: {title}")
    st.write(f"**Authors:** {', '.join(authors)}")
    with st.expander("Show Abstract"):
        st.write(abstract)

    # SUMMARIZATION & INSIGHT EXTRACTION
    if st.button("Generate Summary & Extract Insights"):
        text_to_analyze = abstract if abstract != 'Abstract not Found' else raw_text[:5000]
        
        with st.spinner("Analyzing and Extracting Insights..."):
            # 1. Generate Summary
            summary_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Provide a concise research summary."},
                    {"role": "user", "content": f"Summarize: {text_to_analyze}"}
                ],
                model="llama-3.3-70b-versatile",
            )
            summary_result = summary_completion.choices[0].message.content
            
            st.subheader("📝 Key Research Summary")
            st.write(summary_result)

            # 2. Extract Structured Insights
            insight_prompt = f"""
            You are a research intelligence system. Extract domain-specific insights from this text.
            Return ONLY a valid JSON object with the following keys:
            - "domain" (string)
            - "methods" (list of strings)
            - "datasets" (list of strings)
            - "metrics" (list of strings)
            - "key_findings" (list of strings)
            
            Text: {text_to_analyze}
            """
            
            insight_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": insight_prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1, # Low temperature to ensure strict JSON formatting
                response_format={"type": "json_object"} # Forces Llama to output valid JSON
            )
            
            insights_json_str = insight_completion.choices[0].message.content
            insights_dict = json.loads(insights_json_str)

            st.subheader("🔍 Structured Insights")
            st.json(insights_dict)
            
            # THE DOWNLOAD BUTTON
            st.divider()
            
            # Combine everything into one final payload
            final_data = {
                "metadata": {
                    "title": title,
                    "authors": authors
                },
                "summary": summary_result,
                "insights": insights_dict
            }
            
            st.download_button(
                label="📂 Download Enriched Data (JSON)",
                data=json.dumps(final_data, indent=4),
                file_name=f"{title.replace(' ', '_')}_insights.json",
                mime="application/json"
            )