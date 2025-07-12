from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import requests
import langchain
import re
import base64

langchain.verbose = False
load_dotenv()

# Step 1: Process and vectorize PDF text
def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

# Count words in PDF
def count_words_in_pdf(text):
    return len(text.split())

# Detect if user asked to summarize or explain
def get_prompt_type(query):
    query_lower = query.lower()
    if re.search(r'\bsummar(y|ize|ies|ized)\b', query_lower):
        return "summarize"
    elif re.search(r'\bexplain(s|ed|ation)?\b', query_lower):
        return "explain"
    elif re.search(r'\bwhat is\b|\bdefine\b|\bdefinition\b', query_lower):
        return "define"
    else:
        return "answer"

# Query LLaMA 3.2 via Ollama with improved prompt engineering
def ask_llama(query, context):
    prompt_type = get_prompt_type(query)

    instructions = {
        "summarize": (
            "Provide a concise summary of the key points from the context that are relevant to the query. "
            "Focus only on the most important information. Keep it brief and to the point."
        ),
        "explain": (
            "Explain the concept clearly and in simple terms using only the provided context. "
            "Break it down into easy-to-understand parts. Provide examples if available in the context."
        ),
        "define": (
            "Give a clear and precise definition based on the context. "
            "If the context provides multiple aspects, mention the most relevant ones first."
        ),
        "answer": (
            "Provide a direct and accurate answer to the question using only the context. "
            "If the answer isn't in the context, say 'The answer is not available in the document.' "
            "Do not make up information."
        )
    }

    prompt = f"""You are a helpful assistant that provides accurate information based strictly on the given context.
Follow these instructions carefully:
1. Only use information from the provided context
2. Do not add any information not present in the context
3. If the answer isn't in the context, say so
4. Be concise and stay focused on the query

{instructions[prompt_type]}

Context:
{context}

Query: {query}

Response:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9
                }
            }
        )
        response_data = response.json()
        answer = response_data.get("response", "No response from LLaMA.")
        return answer.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# PDF Preview Function with pinch-to-zoom
def show_pdf(file):
    pdf_bytes = file.getvalue()
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    
    pdf_display = f"""
    <style>
        .pdf-container {{
            width: 100%;
            height: 70vh;
            overflow: auto;
            background: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-top: 10px;
            resize: both;
            min-width: 300px;
            min-height: 400px;
            touch-action: none;
        }}
        .pdf-viewer {{
            width: 100%;
            height: 100%;
            border: none;
            transform-origin: 0 0;
            transition: transform 0.1s ease;
        }}
    </style>
    
    <div class="pdf-container" id="pdf-container">
        <iframe class="pdf-viewer" id="pdf-viewer"
                src="data:application/pdf;base64,{base64_pdf}#toolbar=0&navpanes=0&scrollbar=1">
        </iframe>
    </div>
    
    <script>
        let scale = 1;
        const container = document.getElementById('pdf-container');
        const viewer = document.getElementById('pdf-viewer');
        let lastDistance = 0;
        
        // Touch event handlers for pinch zoom
        container.addEventListener('touchstart', function(e) {{
            if (e.touches.length === 2) {{
                e.preventDefault();
                lastDistance = getDistance(e.touches[0], e.touches[1]);
            }}
        }}, {{ passive: false }});
        
        container.addEventListener('touchmove', function(e) {{
            if (e.touches.length === 2) {{
                e.preventDefault();
                const newDistance = getDistance(e.touches[0], e.touches[1]);
                const delta = newDistance - lastDistance;
                
                if (Math.abs(delta) > 5) {{  // Threshold to prevent jitter
                    if (delta > 0) {{
                        // Pinch out (zoom in)
                        scale = Math.min(scale + 0.02, 3);
                    }} else {{
                        // Pinch in (zoom out)
                        scale = Math.max(scale - 0.02, 0.5);
                    }}
                    viewer.style.transform = `scale(${{scale}})`;
                    lastDistance = newDistance;
                }}
            }}
        }}, {{ passive: false }});
        
        container.addEventListener('touchend', function(e) {{
            lastDistance = 0;
        }});
        
        // Helper function to calculate distance between two touch points
        function getDistance(touch1, touch2) {{
            const dx = touch1.clientX - touch2.clientX;
            const dy = touch1.clientY - touch2.clientY;
            return Math.sqrt(dx * dx + dy * dy);
        }}
        
        // Mouse wheel zoom (for non-touch devices)
        container.addEventListener('wheel', function(e) {{
            if (e.ctrlKey) {{
                e.preventDefault();
                if (e.deltaY < 0) {{
                    scale = Math.min(scale + 0.1, 3);
                }} else {{
                    scale = Math.max(scale - 0.1, 0.5);
                }}
                viewer.style.transform = `scale(${{scale}})`;
            }}
        }}, {{ passive: false }});
    </script>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit App
def main():
    st.set_page_config(page_title="Smart PDF Query", layout="wide")
    st.markdown("""
        <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        .stButton>button {
            background-color: #1e3a8a;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            border: none;
        }
        .stTextInput>div>div>input {
            background-color: black;
        }
        .stFileUploader {
            border: 1px solid #ccc;
            border-radius: 6px;
            padding: 12px;
        }
        /* PDF preview column styling */
        .pdf-preview-column {{
            position: sticky;
            top: 20px;
            height: 85vh;
            overflow: hidden;
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #ddd;
        }}
        @media (max-width: 1200px) {{
            .pdf-preview-column {{
                position: relative;
                height: 60vh;
            }}
        }}
        </style>
    """, unsafe_allow_html=True)

    st.title("Smart PDF Query")

    # Create columns for layout
    col1, col2 = st.columns([6, 4], gap="medium")

    with col1:
        st.subheader("Upload and Query")
        pdf = st.file_uploader("Upload your PDF file", type="pdf", key="pdf_uploader")

        if pdf is not None:
            try:
                # Display PDF processing in the main column
                with st.spinner("Processing PDF..."):
                    # Reset file pointer to beginning in case it was read before
                    pdf.seek(0)
                    pdf_reader = PdfReader(pdf)
                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text

                    knowledgeBase = process_text(text)

                query = st.text_input('Ask a question based on the PDF:')
                cancel_button = st.button('Cancel')

                if cancel_button:
                    st.stop()

                if query:
                    st.write(f"Your query: **{query}**")

                    if "how many words" in query.lower():
                        word_count = count_words_in_pdf(text)
                        st.success(f"The document contains approximately {word_count} words.")
                    else:
                        docs = knowledgeBase.similarity_search(query, k=4)
                        context = "\n\n".join([doc.page_content for doc in docs])

                        with st.expander("Context used for this query"):
                            st.write(context)

                        with st.spinner("Generating response..."):
                            answer = ask_llama(query, context)

                            prompt_type = get_prompt_type(query)
                            label_map = {
                                "summarize": "Summary",
                                "explain": "Explanation",
                                "define": "Definition",
                                "answer": "Answer"
                            }

                            st.subheader(label_map.get(prompt_type, "Response"))
                            st.write(answer)
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

    # PDF Preview Column
    with col2:
        if pdf is not None:
            st.subheader("Document Preview")
            try:
                # Reset file pointer to beginning before showing preview
                pdf.seek(0)
                show_pdf(pdf)
            except Exception as e:
                st.error(f"Error displaying PDF preview: {str(e)}")
                st.info("Please try uploading the file again")

if __name__ == "__main__":
    main()