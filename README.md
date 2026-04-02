# 🤖 RAG Chatbot — eBay User Agreement Q&A System

> A Retrieval-Augmented Generation (RAG) chatbot that answers questions based on the eBay User Agreement document. Built with LLaMA 3.1, FAISS vector database, and Streamlit with real-time streaming responses.

---

## 📽️ Demo

### 🎥 Video Walkthrough (Loom)
> 📌 **[Click here to watch the demo video](#)**


The demo video covers:
- Project architecture and thought process
- How the RAG pipeline works end-to-end
- Live chatbot with streaming responses
- How to run the project locally

### 🖼️ Chatbot Screenshot

> `![Chatbot Screenshot](assets/screenshot.png)`
> <img width="1903" height="1125" alt="image" src="https://github.com/user-attachments/assets/fe0e5359-282b-489b-a7b2-c9865661cc54" />


---

## 🔗 Repository

**GitHub:** [https://github.com/ayushi030709/rag-chatbot](#)


---

## 🧠 Project Overview

This project implements a complete RAG (Retrieval-Augmented Generation) pipeline for question answering over a legal document. Instead of sending the entire 10,500-word eBay User Agreement to an LLM, the system retrieves only the most relevant chunks for each query and grounds the model's response in those excerpts.

### Architecture Flow
```
PDF Document
     ↓
[Preprocessor] → Clean text → Sentence-aware chunks (200 words, 30-word overlap)
     ↓
[Embedder] → all-MiniLM-L6-v2 → 384-dim vectors → FAISS Index
     ↓
[Retriever] → Embed query → Cosine similarity search → Top-4 chunks
     ↓
[Generator] → Prompt injection → LLaMA 3.1 via Groq → Streamed response
     ↓
[Streamlit UI] → Real-time token display + Source passage viewer
```

---

## 🗂️ Folder Structure
```
rag-chatbot/
├── data/
│   └── AI_Training_Document.pdf      ← source document
├── chunks/
│   └── chunks.json                   ← processed text segments
├── vectordb/
│   ├── index.faiss                   ← FAISS vector index
│   └── chunks_meta.pkl               ← chunk text metadata
├── notebooks/
│   └── exploration.ipynb             ← preprocessing experiments
├── src/
│   ├── preprocessor.py               ← PDF extraction + chunking
│   ├── embedder.py                   ← embedding + FAISS storage
│   ├── retriever.py                  ← semantic search
│   └── generator.py                  ← LLM call + streaming
├── app.py                            ← Streamlit chatbot UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Tool | Reason |
|---|---|---|
| LLM | LLaMA 3.1 8B Instruct (via Groq) | Free, fast, instruction-optimized |
| Embedding Model | all-MiniLM-L6-v2 | Free, runs on CPU, semantic similarity |
| Vector Database | FAISS (IndexFlatIP) | Local, no server needed, fast |
| PDF Parsing | PyMuPDF (fitz) | Accurate, preserves reading order |
| UI Framework | Streamlit | Simple, supports streaming |
| Streaming | Python Generators + Groq stream API | Real-time token delivery |

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.9 or higher
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Set Up API Key

Create a `.env` file in the root folder:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free key at [console.groq.com](https://console.groq.com) → API Keys → Create API Key

### Step 4 — Process the Document (Run Once)
```bash
# Extract text, clean, and chunk the PDF
python src/preprocessor.py

# Generate embeddings and build FAISS index
python src/embedder.py
```

### Step 5 — Launch the Chatbot
```bash
python -m streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

---

## 💡 Example Questions to Try
```
What is eBay's arbitration policy?
How do I opt out of arbitration?
What fees does eBay charge sellers?
What is the eBay Money Back Guarantee?
What happens if I don't pay my fees?
Can I sue eBay in court?
What law governs this agreement?
```

---

## 🔍 How the RAG Pipeline Works

### 1. Document Preprocessing (`src/preprocessor.py`)
- Extracts raw text from the PDF using PyMuPDF
- Cleans formatting artifacts (extra whitespace, broken lines)
- Splits text into ~200 word chunks at sentence boundaries
- Applies 30-word overlap between consecutive chunks
- Saves chunks as `chunks/chunks.json`

### 2. Embedding & Indexing (`src/embedder.py`)
- Loads all chunks from JSON
- Converts each chunk to a 384-dimensional vector using `all-MiniLM-L6-v2`
- Normalizes vectors (L2) for cosine similarity comparison
- Stores vectors in a FAISS `IndexFlatIP` index
- Saves index to `vectordb/index.faiss`

### 3. Retrieval (`src/retriever.py`)
- Embeds the user's question using the same model
- Searches FAISS for the top-4 most similar chunk vectors
- Returns matching chunks with their similarity scores

### 4. Generation (`src/generator.py`)
- Injects retrieved chunks + user question into a structured prompt
- Sends prompt to LLaMA 3.1 via Groq API with `stream=True`
- Yields tokens one-by-one using Python generators
- Temperature set to 0.2 for factual, grounded responses

### 5. Streamlit UI (`app.py`)
- Displays streamed tokens in real time with a typing cursor effect
- Shows source passages used to generate each answer
- Sidebar displays model info, chunk count, and example questions
- Clear chat button resets conversation history

---

## 📊 Model & Embedding Choices

### Why `all-MiniLM-L6-v2`?
- Specifically trained for semantic similarity tasks
- 384 dimensions — compact but expressive
- Runs fully on CPU with no GPU needed
- Free and works offline after first download

### Why LLaMA 3.1 via Groq?
- Instruction-optimized model (satisfies fine-tuned LLM requirement)
- Groq provides free API access with fast inference (500+ tokens/sec)
- Supports streaming natively
- Open-source model with strong reasoning capabilities

### Why FAISS?
- No server or Docker setup required
- Saves as a single binary file
- Performs exhaustive search across 68 chunks in under 5ms
- Free and maintained by Meta

---

## 📝 Key Design Decisions

**Chunk size of 200 words:** Legal documents contain complex sentences that need enough context to be meaningful. 200 words fits approximately one complete legal clause or paragraph.

**30-word overlap:** Legal clauses sometimes span paragraph boundaries. Overlapping ensures that boundary information always appears fully in at least one chunk.

**Temperature 0.2:** Low temperature reduces creative/hallucinated responses. For legal Q&A, factual accuracy matters more than varied phrasing.

**Top-4 retrieval:** Provides enough context to answer multi-part questions while keeping the prompt focused and avoiding LLM confusion from too much context.

---

## ⚠️ Known Limitations

- The chatbot only knows about the eBay User Agreement — not other eBay policy pages
- Questions referencing other eBay documents will receive partial answers
- No reranking step — retrieval is based purely on cosine similarity
- Groq free tier has rate limits under heavy usage

---

## 📄 Report

The full 3-page technical PDF report is included in the repository:
`RAG_Chatbot_Report.pdf`

It covers:
- Document structure and chunking logic
- Embedding model and vector database explanation
- Prompt format and generation logic
- 5 example queries (success + failure cases)
- Notes on hallucinations and model limitations

---

## 👩‍💻 Author

**Ayushi**
Junior AI Engineer Assignment — Amlgo Labs — 2026
