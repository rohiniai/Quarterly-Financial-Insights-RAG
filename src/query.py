import gc
import os
import shutil
import time
import torch
import requests
from typing import Sequence, Any, Optional
from operator import itemgetter
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_community.document_transformers import LongContextReorder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks
from langchain_core.documents.compressor import BaseDocumentCompressor
import ingest
from langchain_openai import ChatOpenAI

# 1. Setup API Key 
load_dotenv()

# Automatic device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔌 Using device: {device}")

# Enable GPU for embeddings only if available
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={'device': device}
)

# 1. Initialize the Reorderer
reorderer = LongContextReorder()

# 2. Define a helper function to reorder docs
def reorder_documents(docs):
    return reorderer.transform_documents(docs)

def load_system_prompt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Custom OpenRouter Reranker Implementation
class OpenRouterRerank(BaseDocumentCompressor):
    model: str = "cohere/rerank-4-pro"
    top_n: int = 12
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY")

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        
        url = "https://openrouter.ai/api/v1/rerank"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare documents for the API
        doc_list = [doc.page_content for doc in documents]
        
        payload = {
            "model": self.model,
            "query": query,
            "documents": doc_list,
            "top_n": self.top_n
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            print(f"❌ Rerank Error: {response.status_code} - {response.text}")
            return documents[:self.top_n] # Fallback to top_n original
            
        data = response.json()
        
        # OpenRouter/Cohere format: results: [{index: 0, relevance_score: 0.9}, ...]
        results = data.get("results", [])
        
        final_docs = []
        for res in results:
            idx = res.get("index")
            if idx is not None and idx < len(documents):
                doc = documents[idx]
                doc.metadata["relevance_score"] = res.get("relevance_score")
                final_docs.append(doc)
                
        return final_docs

def setup_rag_chain():
    print("🧠 Loading AI Brain...")

    # 0. Setup LLM for final answer using OpenRouter
    llm = ChatOpenAI(
        model="google/gemini-2.0-flash-001",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0
    )
    
    # Identify all PDF sources - handle paths relative to project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data", "10Q")
    root_dir = base_dir
    
    # Get PDFs from data/10Q
    pdf_files_data = []
    if os.path.exists(data_dir):
        pdf_files_data = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]
    
    all_pdf_paths = pdf_files_data
    
    if not all_pdf_paths:
        print("⚠️ No PDF files found! Please check your data directories.")
        return None
        
    folder_sources = {os.path.basename(f).split("_")[0].upper() for f in all_pdf_paths}
    
    # Check if chromadb exists and has all the data
    should_reprocess = True
    chroma_path = os.path.join(base_dir, "chromadb")
    db_file = os.path.join(chroma_path, "chroma.sqlite3")
    
    if os.path.exists(db_file):
        print("📦 Checking existing database via sqlite...")
        import sqlite3
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT string_value FROM embedding_metadata WHERE key='source';")
            db_sources = {row[0] for row in cursor.fetchall() if row[0]}
            conn.close()
            
            if folder_sources.issubset(db_sources) and len(db_sources) > 0:
                print(f"✅ Database already contains data for: {', '.join(db_sources)}")
                should_reprocess = False
                vectorstore = Chroma(
                    persist_directory=chroma_path,
                    embedding_function=embeddings
                )
            else:
                print(f"⚠️ Database incomplete or missing. Expected {folder_sources}, found {db_sources}. Re-ingesting...")
        except Exception as e:
            print(f"⚠️ Error checking database: {e}. Re-ingesting...")
    
    if should_reprocess:
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            
        print(f"🚀 Starting fresh ingestion of {len(all_pdf_paths)} files...")
        vectorstore = ingest.process_pdfs_from_list(all_pdf_paths)
    
    # 2. Setup Hybrid Search
    all_content = vectorstore.get()
    
    # Efficient keyword search
    keyword_retriever = BM25Retriever.from_texts(
        all_content['documents'], 
        metadatas=all_content['metadatas']
    )
    keyword_retriever.k = 25  
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

    # Hybrid blend
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, keyword_retriever], 
        weights=[0.4, 0.6] 
    )

    # 3. Using Custom OpenRouter Reranker
    compressor = OpenRouterRerank(
        model="cohere/rerank-4-pro", 
        top_n=12
    )
    
    # Wrap the hybrid retriever with the reranker
    from langchain_classic.retrievers import ContextualCompressionRetriever
    reranked_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=hybrid_retriever
    )

    # 4. Multi-Company Strategy
    from langchain_core.runnables import RunnableLambda
    
    # 🔍 DYNAMIC COMPANY DETECTION
    db_file = os.path.join(base_dir, "chromadb", "chroma.sqlite3")
    cached_db_sources = []
    if os.path.exists(db_file):
        import sqlite3
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            # Try to get unique sources from the DB
            cursor.execute("SELECT DISTINCT string_value FROM embedding_metadata WHERE key='source';")
            cached_db_sources = [row[0].upper() for row in cursor.fetchall() if row[0]]
            conn.close()
        except Exception:
            pass
    
    # Fallback to file system if DB is empty/unavailable
    if not cached_db_sources:
        pdf_dir = os.path.join(base_dir, "data", "10Q")
        if os.path.exists(pdf_dir):
            cached_db_sources = [f.split("_")[0].upper() for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    def multi_company_search(query_input):
        q_upper = query_input.upper()
        
        # FIX: Robust detection. Check if the query contains parts of the source name
        # OR if the source name contains words from the query.
        found_companies = []
        for source in cached_db_sources:
            source_upper = source.upper()
            # Clean source (e.g., "ELI_LILLY" -> "LILLY") to check against query
            short_source = source_upper.split('_')[0] 
            
            if source_upper in q_upper or short_source in q_upper:
                if source not in found_companies:
                    found_companies.append(source)

        # Generic comparative keywords
        compare_keywords = ["ALL", "EACH", "COMPARE", "DIFFERENCE", "MOST", "LEAST", "HIGHEST", "LOWEST"]
        if any(k in q_upper for k in compare_keywords) and not found_companies:
            found_companies = cached_db_sources

        if not found_companies:
            return reranked_retriever.invoke(query_input)
            
        print(f"🏢 Targeted search for: {', '.join(found_companies)}")
        
        all_final_candidates = []
        CHUNKS_PER_COMPANY = 6 

        for company in found_companies:
            print(f"  -> Gathering dedicated data for {company}...")
            
            # Use the EXACT source tag for the metadata filter
            company_filter = {"source": company}
            
            # 1. Vector Search
            vector_docs = vectorstore.similarity_search(query_input, k=12, filter=company_filter)
            
            # 2. Keyword Nudge (Searching for "BERKSHIRE_HATHWAY Nasdaq")
            kw_docs = keyword_retriever.invoke(f"{company} {query_input}")
            filtered_kw = [d for d in kw_docs if d.metadata.get("source") == company]
            
            # Combine and locally rerank to find the best info for THIS specific company
            company_pool = vector_docs + filtered_kw
            if company_pool:
                reranked_company_pool = compressor.compress_documents(company_pool, query_input)
                all_final_candidates.extend(reranked_company_pool[:CHUNKS_PER_COMPANY])

        # Deduplicate
        seen = set()
        unique_docs = []
        for doc in all_final_candidates:
            content_id = f"{doc.metadata.get('source')}_{doc.page_content[:50]}"
            if content_id not in seen:
                unique_docs.append(doc)
                seen.add(content_id)
                
        return unique_docs
    system_prompt = load_system_prompt('src/system_prompt.md') if os.path.exists('src/system_prompt.md') else load_system_prompt('system_prompt.md')

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Question: {input}"),
    ])
    
    # Create the modern retrieval chain
    rag_chain = (
        {
            "context": itemgetter("input") | RunnableLambda(multi_company_search) | reorder_documents, 
            "input": itemgetter("input")
        }
        | RunnablePassthrough.assign(
            answer=(
                prompt 
                | llm 
                | StrOutputParser()
            )
        )
    )
    
    return rag_chain

def ask_finance(rag_chain, question):
    print(f"\n🔍 Searching 10-Qs for: {question}")
    
    # 1. Get the result from the chain
    result = rag_chain.invoke({"input": question})
    
    # 2. Print the answer
    print("\n" + "🤖 AI ANSWER:" + "\n" + "-"*15)
    print(result['answer'].strip())
    
    # 3. Print the sources
    print("\n📚 SOURCES USED:")
    sources = sorted(list(set([
        f"- {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', '?')})" 
        for doc in result['context']
    ])))
    
    for s in sources:
        print(s)
        
    return result

if __name__ == "__main__":
    chain = setup_rag_chain()
    
    print("\n" + "="*50)
    print("🚀 FINANCIAL RAG TERMINAL READY")
    print("="*50)

    while True:
        print("\n" + "—"*50)
        user_query = input("💬 Question: ").strip()

        if user_query.lower() in ['exit', 'quit', 'q']:
            print("👋 Closing session.")
            break
        
        if user_query:
            ask_finance(chain, user_query)
        else:
            print("⚠️ Please enter a question.")
