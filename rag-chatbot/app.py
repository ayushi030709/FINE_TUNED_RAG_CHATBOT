# app.py
import streamlit as st
from src.retriever import Retriever
from src.generator import stream_response

# ── Page Configuration ────────────────────────────────────
st.set_page_config(
    page_title="eBay Agreement Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS for better look ────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #e53238, #f5af02);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        border-left: 4px solid #e53238;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Retriever once (cached) ─────────────────────────
@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title(" System Info")
    st.markdown("---")
    
    # System details
    st.markdown("** LLM Model**")
    st.code("llama-3.1-8b-instant")
    
    st.markdown("** Embedding Model**")
    st.code("all-MiniLM-L6-v2")
    
    st.markdown("** Vector Database**")
    st.code("FAISS (IndexFlatIP)")
    
    st.markdown("** Indexed Chunks**")
    st.code(f"{retriever.index.ntotal} chunks")
    
    st.markdown("** Top-K Retrieval**")
    st.code("4 chunks per query")
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**💡 Try These Questions:**")
    
    example_questions = [
        "What is the arbitration policy?",
        "How do I opt out of arbitration?",
        "What fees does eBay charge sellers?",
        "What is the eBay Money Back Guarantee?",
        "What happens if I don't pay my fees?",
        "Can I sue eBay in court?",
        "What law governs this agreement?"
    ]
    
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q
            st.rerun()

# ── Main Area ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 eBay User Agreement Chatbot</h1>
    <p>Ask any question about the eBay User Agreement — powered by RAG + LLaMA 3</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📎 View Source Passages"):
                for i, chunk in enumerate(message["sources"]):
                    st.markdown(f"""
                    <div class="source-box">
                    <b>Excerpt {i+1}</b> | Relevance Score: {chunk['score']:.3f}<br><br>
                    {chunk['text'][:350]}...
                    </div>
                    """, unsafe_allow_html=True)

# Handle sidebar button clicks
if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    user_input = st.chat_input("Ask anything about the eBay User Agreement...")

# Process user input
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant chunks
    with st.spinner("🔍 Searching document for relevant sections..."):
        chunks = retriever.retrieve(user_input, top_k=4)

    # Stream the response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Stream tokens one by one
        for token in stream_response(user_input, chunks):
            full_response += token
            response_placeholder.markdown(full_response + "▌")  # blinking cursor effect

        # Final response without cursor
        response_placeholder.markdown(full_response)

        # Show source passages
        with st.expander("📎 View Source Passages Used"):
            for i, chunk in enumerate(chunks):
                st.markdown(f"""
                <div class="source-box">
                <b>Excerpt {i+1}</b> | Relevance Score: {chunk['score']:.3f}<br><br>
                {chunk['text'][:350]}...
                </div>
                """, unsafe_allow_html=True)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": chunks
    })
