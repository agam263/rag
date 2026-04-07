import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables (pulls GOOGLE_API_KEY from .env)
load_dotenv()

# Verify securely that the GOOGLE_API_KEY is present
if not os.environ.get("GOOGLE_API_KEY"):
    # Streamlit Cloud deployment fail-safe reading Native Secrets
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    else:
        st.error("GOOGLE_API_KEY missing from environment. Please configure it in your Streamlit Cloud Secrets or local .env file.")
        st.stop()

# ---------------------------- #
# Frontend Configuration
# ---------------------------- #
st.set_page_config(page_title="RAG Document Assistant", layout="centered", page_icon="✨")

# High-End Premium UI/UX Implementation (CSS Injection)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

/* Base Font & Theme Configuration */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* Deep Black Superhero Theme */
.stApp, .appview-container, .main {
    background: #050505 !important;
}

/* Floating Superhero Bubbles Container */
.bubbles-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 0;
    overflow: hidden;
    pointer-events: none; /* Ignore clicks */
}

/* Base style for individual bubbles */
.bubble {
    position: absolute;
    bottom: -150px;
    opacity: 0;
    filter: drop-shadow(0 0 15px rgba(255,255,255,0.15));
    animation: floatUp infinite ease-in;
}

@keyframes floatUp {
    0% { transform: translateY(0) rotate(0deg); opacity: 0; }
    15% { opacity: 0.6; }
    85% { opacity: 0.6; }
    100% { transform: translateY(-120vh) rotate(360deg); opacity: 0; }
}

/* Glassmorphism Effect for the Main Container */
.main .block-container {
    position: relative;
    z-index: 10;
    background: rgba(25, 25, 25, 0.45);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 24px;
    padding: 3rem !important;
    margin-top: 3rem;
    box-shadow: 0 15px 50px rgba(0, 0, 0, 0.7);
    animation: containerEntrance 1s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
    transform: translateY(30px);
    opacity: 0;
}

@keyframes containerEntrance {
    to { transform: translateY(0); opacity: 1; }
}

/* Stunning Title Glow & Animation */
h1 {
    text-align: center;
    background: linear-gradient(135deg, #c084fc, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 3rem !important;
    margin-bottom: 0.5rem !important;
    animation: glowPulse 3s infinite alternate;
}

@keyframes glowPulse {
    from { text-shadow: 0 0 10px rgba(139, 92, 246, 0.1); }
    to { text-shadow: 0 0 20px rgba(56, 189, 248, 0.3); }
}

/* Interactive File Uploader */
[data-testid="stFileUploadDropzone"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 2px dashed rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(139, 92, 246, 0.05) !important;
    border-color: #c084fc !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.15);
}

/* Input Fields */
input {
    background: rgba(0, 0, 0, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    color: white !important;
    transition: all 0.3s ease !important;
}

input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.2) !important;
}

/* Gradient Action Buttons with Zoom & Glow */
[data-testid="baseButton-secondary"] {
    background: linear-gradient(135deg, #6366f1, #0ea5e9) !important;
    border: none !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 0.5rem 2rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}

[data-testid="baseButton-secondary"]:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 10px 25px rgba(14, 165, 233, 0.4) !important;
    filter: brightness(1.1) !important;
}

/* Hide Unnecessary Streamlit Elements */
header { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

import random

# Generate 50 Floating Superhero Bubbles
emojis = ["🦸‍♂️", "🦇", "🕷️", "🤖", "⚡", "🛡️", "💥", "🦸‍♀️"]
bubbles_html = '<div class="bubbles-container">'
for _ in range(50):
    emoji = random.choice(emojis)
    left = random.uniform(0, 100)
    size = random.uniform(20, 55)
    anim_duration = random.uniform(10, 25)
    anim_delay = random.uniform(0, 20)
    
    bubbles_html += f'<div class="bubble" style="left: {left}%; font-size: {size}px; animation-duration: {anim_duration}s; animation-delay: {anim_delay}s;">{emoji}</div>'
bubbles_html += '</div>'

# Inject bubbles into the background securely
st.markdown(bubbles_html, unsafe_allow_html=True)

# Title of the Streamlit App
st.markdown("<h1 style='position: relative; z-index: 10;'>✨ Nexus RAG Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.6); margin-bottom: 2rem; position: relative; z-index: 10;'>Unlock knowledge from your documents using AI and Embeddings</p>", unsafe_allow_html=True)

# Initialize session state variables so the app remembers state between user interactions
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = ""

# File uploader to upload a PDF document
uploaded_file = st.file_uploader("Upload a PDF document to begin", type=["pdf"])

# ---------------------------- #
# Document Processing
# ---------------------------- #
if uploaded_file is not None:
    # We only process the file if it hasn't been processed yet on this session
    if st.session_state.processed_filename != uploaded_file.name:
        st.session_state.qa_chain = None  # Reset pipeline for new doc
        tmp_path = None
        
        with st.spinner("Analyzing and parsing the document..."):
            try:
                # Save uploaded file temporarily for PyPDFLoader to read
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                    
                # 1. Load the document
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                # 2. Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                # 3. Create HuggingFace Embeddings (Free, Local Alternative)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                # 4. Store them in Chroma vector database securely
                persist_dir = "chroma_data_streamlit"
                vector_db = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=persist_dir
                )
                
                # Set up the LLM (Google Gemini Flash) and Retriever mechanism
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
                retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                
                # Construct RetrievalQA Chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                
                # Persist the chain state uniquely
                st.session_state.qa_chain = qa_chain
                st.session_state.processed_filename = uploaded_file.name
                st.success("✅ File securely processed and vectorized!")
                
            except Exception as e:
                st.error(f"Error processing document: {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

# ---------------------------- #
# Chat / Generation Mechanism
# ---------------------------- #
# Text input box to enter a question
question = st.text_input("Ask a question based on your document:")

# Button labeled "Ask"
if st.button("Ask"):
    if st.session_state.qa_chain is None:
        st.warning("Please upload a valid document first.")
    elif not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Querying the Gemini LLM Sandbox..."):
            try:
                # 5. Retrieve relevant chunks and Send context to Gemini
                response = st.session_state.qa_chain.invoke({"query": question})
                answer = response.get("result", "No answer generated.")
                source_docs = response.get("source_documents", [])
                
                # 6. Area to display the answer
                st.markdown("### 🤖 Answer")
                st.write(answer)
                
                # Extra: Expandable box to view where the answer came from
                with st.expander("View Source Context"):
                    for idx, doc in enumerate(source_docs, 1):
                        st.markdown(f"**Chunk {idx}:**")
                        st.write(doc.page_content)
                        st.divider()
                        
            except Exception as e:
                st.error(f"An error occurred over generation: {e}")
