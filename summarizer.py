import os
from dotenv import load_dotenv
from groq import Groq

# 1. Load variables from .env [cite: 146, 149, 150]
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 2. Check if the key actually loaded
if not api_key:
    print("❌ ERROR: GROQ_API_KEY not found in .env file!")
else:
    print("✅ API Key loaded successfully.")
    client = Groq(api_key=api_key)

    def generate_research_summary(text_path):
        if not os.path.exists(text_path):
            return f"❌ Error: {text_path} not found."

        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read()

        print("Sending text to Groq...")
        # Using Llama 3 as suggested in your project plan [cite: 21, 104]
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": f"Summarize this research: {content[:8000]}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3 # Low temperature for factual output [cite: 142, 143]
        )
        return chat_completion.choices[0].message.content

    if __name__ == "__main__":
        result = generate_research_summary("parsed_output/extracted_text.txt")
        print("\n--- SUMMARY ---\n")
        print(result)