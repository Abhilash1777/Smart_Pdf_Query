# 📚 Smart PDF Query

**An intelligent PDF analyzer with semantic search and context-aware responses**  
Query your documents like never before - get summaries, explanations, and precise answers powered by LLaMA 3 through Ollama.

![Demo](https://via.placeholder.com/800x500?text=Smart+PDF+Query+Demo) *Replace with actual demo GIF*

## ✨ Key Features

- **Semantic Document Understanding**  
  FAISS vector embeddings capture document meaning beyond keyword matching

- **Context-Aware Responses**  
  Automatically detects when you want summaries, explanations, or direct answers

- **Advanced PDF Handling**  
  - Multi-page text extraction  
  - Interactive pinch-to-zoom preview  
  - Word count analysis

- **Optimized AI Prompts**  
  Precision-engineered instructions for LLaMA 3 to ensure accurate, context-bound responses

- **Professional UI**  
  Responsive two-panel layout with dark mode support and intuitive controls

## 🛠️ Tech Stack

- **Backend**: Python 3.10+
- **AI Framework**: LangChain
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **Vector Store**: FAISS
- **LLM**: LLaMA 3 via Ollama (local)
- **Frontend**: Streamlit
- **PDF Processing**: PyPDF2

## 🚀 Quick Start

### Prerequisites
- Ollama running locally with LLaMA 3 installed
- Python 3.10+

```bash
# Install Ollama (if needed)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
```
### Installation

```
git clone https://github.com/yourusername/smart-pdf-query.git
cd smart-pdf-query

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

pip install -r requirements.txt

```
### Running the App

```
streamlit run app.py
```
## 🧠 How It Works

### Document Processing:

Extracts text from PDFs with PyPDF2

Splits content into semantic chunks

Generates embeddings using HuggingFace

### Query Handling:

Classifies user intent (summarize/explain/answer)

Retrieves relevant document sections

Crafts optimized prompts for LLaMA 3

### Response Generation:

Processes through local Ollama instance

Returns context-bound answers

Preserves document accuracy

## 📂 Project Structure

```
smart-pdf-query/
├── app.py                 # Main application logic
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .env                   # Environment variables
```
## 🌐 Browser Support
✅ Chrome
✅ Edge
✅ Firefox
⚠️ Safari (PDF preview may vary)

## 📜 License
MIT License - Free for personal and commercial use

## ✉️ Contribution
Give your valuable contribution to this project to move it forward with technology.
#
<p align="center" style="color:#7b1fa2;">
  Created with 💙 by Abhilash using React & pure creativity.
</p>
