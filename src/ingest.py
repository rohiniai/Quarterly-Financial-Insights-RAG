import os
import pdfplumber
import gc
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup Tech Stack Components
#mixedbread-ai/mxbai-embed-large-v1
#BAAI/bge-base-en-v1.5 or BAAI/bge-large-en-v1.5
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
text_splitter = RecursiveCharacterTextSplitter(
    # Granular chunks (1000 characters) for better fact extraction
    chunk_size=1000, 
    # 20% overlap to maintain context across headers
    chunk_overlap=200, 
    # Prioritize double newlines to keep sections/tables whole
    separators=["\n\n", "\n", " ", ""]
)

def process_pdfs_from_list(pdf_paths):
    all_chunks = []

    for path in tqdm(pdf_paths, desc="📁 Overall Progress"):
        filename = os.path.basename(path)
        # Clean company name: e.g., "APPLE_Annual_Report.pdf" -> "APPLE"
        company_name = filename.split("_Quaterly")[0].upper()
        print(f"  📄 Processing {company_name}...")

        with pdfplumber.open(path) as pdf:
            # We iterate through pages and chunk them INDIVIDUALLY
            for page_idx, page in enumerate(tqdm(pdf.pages, desc=f"📄 {company_name}", leave=False)):
                text = page.extract_text()
                if text:
                    # Create chunks for THIS specific page
                    page_chunks = text_splitter.create_documents(
                        [text], 
                        metadatas=[{
                            "source": company_name, 
                            "type": "10-Q",
                            "page": page_idx + 1  # Adding 1 because PDF index starts at 0
                        }]
                    )
                    all_chunks.extend(page_chunks)
                # Free memory
                page.flush_cache()

        # Explicit memory cleanup after each file
        gc.collect()

    print(f"\n🚀 Total chunks created: {len(all_chunks)}")
    print("🚀 Starting Vector Encoding (ChromaDB)...")

    # Note: If you are re-running this, you might want to delete the old folder
    # to avoid duplicate entries.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    persist_dir = os.path.join(base_dir, "chromadb")
    
    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"✅ Success! Indexed {len(all_chunks)} chunks with page metadata.")
    return vector_db

def process_pdfs(directory):
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".pdf")]
    return process_pdfs_from_list(pdf_files)


# if __name__ == "__main__":
#     data_path = "./data/10Q/" if os.path.exists("./data/10Q/") else "../data/10Q/"
#     process_pdfs(data_path)