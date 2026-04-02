# src/generator.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Instructions for the LLM
# src/generator.py

SYSTEM_PROMPT = """You are a specialized legal document assistant, 
fine-tuned to answer questions about the eBay User Agreement only.

You have been trained with the following behavioral rules:
1. NEVER answer from general knowledge — only from provided excerpts
2. Always cite which section your answer comes from (e.g. "According to Section 6...")
3. If asked something outside the document, respond: 
   "This information is not covered in the eBay User Agreement."
4. Structure answers clearly with short paragraphs
5. For legal terms, provide plain English explanations
6. Always end with: "Please consult a legal professional for official advice."

You are an expert on:
- eBay fees and seller policies
- Arbitration and legal dispute resolution  
- Buyer and seller protections
- Payment and return policies
- eBay account rules and restrictions"""

def build_prompt(query: str, chunks: list) -> str:
    """
    Combine retrieved chunks + user question into one prompt.
    This is called 'prompt injection' or 'context stuffing'.
    """
    # Join all chunks with separators
    context = "\n\n---\n\n".join([
        f"[Document Excerpt {i+1}]:\n{chunk['text']}"
        for i, chunk in enumerate(chunks)
    ])

    return f"""Below are relevant excerpts from the eBay User Agreement document:

{context}

---

User's Question: {query}

Please answer the question using ONLY the excerpts above:"""


def stream_response(query: str, chunks: list):
    """
    Send prompt to Groq and stream response token by token.
    This is a Python generator — yields one piece of text at a time.
    """
    prompt = build_prompt(query, chunks)

    # Call Groq API with stream=True
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # free model on Groq
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        stream=True,          # ← enables streaming
        max_tokens=1000,
        temperature=0.2       # low = more factual, less creative
    )

    # Yield each token as it arrives
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta