import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import openai
import re
import base64

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

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

# Ask OpenAI instead of Ollama

def ask_openai(query, context):
    prompt_type = get_prompt_type(query)
    instructions = {
        "summarize": "Summarize the content clearly and concisely.",
        "explain": "Explain the concept in simple terms using only the context.",
        "define": "Give a precise definition using only the context.",
        "answer": "Answer the question strictly based on the context provided. Say 'Not in document' if not found."
    }
    prompt = f"""You are a helpful assistant.\nUse only the information from the context below.\n\n{instructions[prompt_type]}\n\nContext:\n{context}\n\nQuery:\n{query}\n\nAnswer:"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# PDF Preview Function
def show_pdf(file):
    pdf_bytes = file.getvalue()
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
    pdf_display = f"""
    <iframe width='100%' height='600' src="data:application/pdf;base64,{base64_pdf}" type="application/pdf" frameborder="0"></iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit App
def main():
    st.set_page_config(page_title="Smart PDF Query", layout="wide")
    st.title("Smart PDF Query")

    col1, col2 = st.columns([6, 4], gap="medium")

    with col1:
        st.subheader("Upload and Query")
        pdf = st.file_uploader("Upload your PDF file", type="pdf", key="pdf_uploader")

        if pdf is not None:
            try:
                with st.spinner("Processing PDF..."):
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
                            answer = ask_openai(query, context)

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

    with col2:
        if pdf is not None:
            st.subheader("Document Preview")
            try:
                pdf.seek(0)
                show_pdf(pdf)
            except Exception as e:
                st.error(f"Error displaying PDF preview: {str(e)}")
                st.info("Please try uploading the file again")

if __name__ == "__main__":
    main()
