import os
import json
from dotenv import load_dotenv
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Load environment variables and initialize Groq Client [cite: 37, 39]
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file!")
    exit()

client = Groq(api_key=api_key)

def ask_research_assistant(question, retrieved_context):
    """
    Improved, user-friendly prompt for RAG-based research queries.
    This fulfills the task of creating a 'Research Intelligence System' persona.
    """
    # The 'system_message' sets the professional persona and structural rules 
    system_message = """
    You are an Expert Research Intelligence System. Your goal is to help users 
    understand complex research papers by providing clear, structured, and cited answers.
    
    GUIDELINES:
    - CLARITY: Use bold headers and bullet points for better readability.
    - CITATIONS: Always mention the paper title or authors when providing specific facts.
    - TERMINOLOGY: If you use a technical term, explain it simply in parentheses.
    - LIMITS: Use ONLY the provided context. If the answer isn't there, say: 
      "I'm sorry, my current database doesn't have details on that specific topic."
    """

    user_message = f"""
    RESEARCH DATABASE CONTEXT FROM VECTOR STORE:
    {retrieved_context}

    USER QUESTION:
    {question}

    Please provide a comprehensive and user-friendly response:
    """

    # Low temperature (0.2) ensures the response is factual and grounded 
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        model="llama-3.3-70b-versatile",
        temperature=0.2 
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("--- 📄 Research Assistant RAG System ---")
    
    # 2. Initialize the Embedding Model (Must match upload_on_RAG.py) [cite: 407, 412]
    # This turns your text into numbers (vectors) for comparison [cite: 409, 432]
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # 3. Load the Local Vector Database [cite: 421, 436]
    # The Vector Store finds the closest matches to your question [cite: 426, 433]
    try:
        if not os.path.exists("research_papers_faiss"):
            print("❌ Error: 'research_papers_faiss' folder not found. Run 'python upload_on_RAG.py' first!")
        else:
            vector_db = FAISS.load_local("research_papers_faiss", embeddings, allow_dangerous_deserialization=True)
            
            # 4. Get User Query
            query = input("\n🔍 What research topic are you interested in? ")

            # 5. Retrieval: Search for top 3 most relevant paper chunks [cite: 419, 420, 433]
            print("\nSearching database...")
            docs = vector_db.similarity_search(query, k=3)
            context = "\n---\n".join([d.page_content for d in docs])

            print(context)

            # 6. Generation: Pass context to the LLM [cite: 434, 437]
            print("🚀 Generating professional response...\n")
            answer = ask_research_assistant(query, context)
            
            print("="*50)
            print("AI RESEARCH ASSISTANT RESPONSE")
            print("="*50)
            #print(answer)
            print("="*50)

    except Exception as e:
        print(f"❌ An error occurred: {e}")