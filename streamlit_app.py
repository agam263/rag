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

# Verify that the GOOGLE_API_KEY is present
if not os.environ.get("GOOGLE_API_KEY"):
    st.error("GOOGLE_API_KEY missing from environment. Please add it to your .env file.")
    st.stop()

# ---------------------------- #
# Frontend Configuration
# ---------------------------- #
st.set_page_config(page_title="RAG Document Assistant", layout="centered")

# Title of the Streamlit App
st.title("RAG Document Assistant")

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
