# 📊 AI Financial Analyst: Advanced 10-Q RAG System

An enterprise-grade Retrieval-Augmented Generation (RAG) system designed for high-precision data extraction and comparative analysis of SEC 10-Q. This system leverages a multi-stage retrieval pipeline and the latest **Gemini 2.0 Flash** model to provide accurate, cited financial insights.

## 🚀 Key Features

*   **Advanced RAG Pipeline**: Uses a hybrid search approach (Vector + BM25) followed by a **Cohere Rerank-4-Pro** stage for maximum precision.
*   **Gemini 2.0 Flash Integration**: Optimized for high-speed, high-reasoning financial data extraction.
*   **Multi-Company Logic**: Automatically detects and separates data for different companies (e.g., Apple vs. Eli Lilly) even within complex, multi-document queries.
*   **Automated Evaluation**: Built-in evaluation suite using **Ragas** to measure Faithfulness, Answer Relevancy, Context Recall, and Answer Correctness.
*   **Financial Precision**: Enforces strict "Senior Financial Analyst" persona via advanced prompt engineering, ensuring data unit consistency and mandatory source attribution.

## 🛠️ Tech Stack

- **LLM**: Gemini 2.0 Flash (via OpenRouter)
- **Framework**: LangChain & LangGraph
- **Vector DB**: ChromaDB
- **Embeddings**: BAAI/bge-base-en-v1.5 (Local GPU/CPU)
- **Reranker**: Cohere Rerank-4-Pro
- **Evaluation**: Ragas
- **PDF Extraction**: pdfplumber

## 📁 Project Structure

```text
├── data/
│   ├── 10Q/                # Store your PDF reports here
│   ├── golden_dataset.json  # Ground truth for evaluation
│   └── eval_ragas_results.json
├── src/
│   ├── ingest.py           # Document processing and vectorization
│   ├── query.py            # Main RAG execution and Terminal UI
│   ├── evaluate.py         # Ragas evaluation suite
│   └── system_prompt.md    # Expert analyst instructions
├── requirements.txt
└── .env                    # API Keys (OpenRouter)
```

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Financial_Project_10Q.git
   cd Financial_Project_10Q
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   OPENROUTER_API_KEY=your_key_here
   ```

## 🔍 Usage

### 💬 Chat with your Documents
Run the interactive terminal to query your financial data:
```bash
python src/query.py
```

### ⚖️ Run Evaluation
Measure the system's performance against the golden dataset:
```bash
python src/evaluate.py
```

## 📊 Evaluation Metrics (Sample Results)
| Metric | Score |
| :--- | :--- |
| **Faithfulness** | 0.8872 |
| **Answer Relevancy** | 0.9102 |
| **Context Recall** | 0.9615 |
| **Answer Correctness** | 0.8245* |
*\*After applying formatting alignment and Gemini 2.0 Flash upgrades.*

## 🛡️ Security & Integrity
This system is designed to work ONLY with provided context. It includes strict guardrails to prevent hallucinations and ensures that every financial figure is attributed to a specific page and document.

---
*Created by [Rohini Ari]*
