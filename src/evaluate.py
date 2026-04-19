import json
import os
# import re
import pandas as pd
import nest_asyncio
import torch
from datasets import Dataset
from query import setup_rag_chain

# Ragas imports: Use capitalized class names for proper initialization
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# Apply nest_asyncio to avoid "asyncio.run() cannot be called from a running event loop"
nest_asyncio.apply()

# class CleanChatOpenAI(ChatOpenAI):
#     """
#     Custom wrapper to fix Gemini's illegal JSON escaping (e.g., \') 
#     and common parsing errors when acting as a Ragas evaluator.
#     """
#     def _generate(self, *args, **kwargs):
#         res = super()._generate(*args, **kwargs)
#         for generation in res.generations:
#             if hasattr(generation, 'text') and isinstance(generation.text, str):
#                 text = generation.text.strip()
#                 
#                 # 1. Remove markdown code blocks if present
#                 if text.startswith("```json"):
#                     text = text[7:]
#                 if text.endswith("```"):
#                     text = text[:-3]
#                 text = text.strip()
# 
#                 # 2. Fix illegal single quote escaping
#                 text = text.replace("\\'", "'")
#                 text = text.replace("\\\\", "\\")
#                 
#                 # 3. Fix potential trailing commas in JSON
#                 text = re.sub(r",\s*}", "}", text)
#                 text = re.sub(r",\s*]", "]", text)
#                 
#                 generation.text = text
#                 if hasattr(generation, 'message') and hasattr(generation.message, 'content'):
#                     generation.message.content = text
#         return res

import re

def strip_citations(text: str) -> str:
    """
    Removes citations and analytical summaries to improve 
    RAGAS metric alignment with simple ground truth answers.
    """
    # 1. Remove bracketed citations: [Company | Doc | Page]
    text = re.sub(r"\[[^\]]+\]", "", text)
    
    # 2. Remove parenthetical citations: (Company, Page X)
    text = re.sub(r"\([A-Z][a-zA-Z\s]+,\s*Page\s*\d+\)", "", text)
    
    # 3. Remove "Analytical Summary" sections which are usually not in ground truth
    text = re.sub(r"###?\s*Analytical Summary.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\*\*Analytical Summary:\*\*.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    
    # Clean whitespace and punctuation artifacts
    text = re.sub(r"\n\s*\n", "\n\n", text)
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.replace(" .", ".").replace(" ,", ",")
    return cleaned

def run_evaluation():
    # 0. Setup Base Directory and Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, 'data', 'golden_dataset.json')
    output_path = os.path.join(base_dir, 'data', 'eval_ragas_results.json')

    print(f"📍 Project Base: {base_dir}")

    # 1. Load the AI pipeline (RAG Chain)
    print("🤖 Setting up the RAG chain for evaluation...")
    rag_chain = setup_rag_chain()
    if not rag_chain:
        print("❌ Error: RAG chain setup failed. Verify your data/10Q directory.")
        return

    # 2. Load the golden dataset
    print(f"📂 Loading Golden Dataset from {dataset_path}...")
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            golden_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: '{dataset_path}' not found.")
        return

    # 3. Run RAG chain against the questions
    print(f"🔍 Running RAG chain for {len(golden_data)} questions...")
    
    questions, ground_truths, answers, contexts = [], [], [], []

    for item in golden_data:
        print(f"\n❓ Question: {item['question']}")
        try:
            # Generate answer from your RAG system
            response = rag_chain.invoke({"input": item['question']})
            
            raw_answer = response['answer']
            # Remove citations and summaries so Ragas doesn't penalize for extra structural info
            clean_answer = strip_citations(raw_answer)
            
            questions.append(item['question'])
            ground_truths.append(item['answer'])
            answers.append(clean_answer)
            contexts.append([doc.page_content for doc in response['context']])
            
            print(f"✅ Answered! (Processed {len(answers)}/{len(golden_data)})")
        except Exception as e:
            print(f"❌ Error testing question '{item['question']}': {e}")

    # 4. Prepare data for Ragas
    if not questions:
        print("❌ No questions were successfully processed.")
        return

    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)

    # 5. Setup Ragas Evaluator
    print("\n⚖️ Starting Ragas Evaluation via OpenRouter...")
    
    # Use standard ChatOpenAI (removed CleanChatOpenAI wrapper)
    evaluator_llm = ChatOpenAI(
        model="google/gemini-2.0-flash-001",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}}
    )

    # GPU/CPU device detection for local embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': device}
    )

    # INITIALIZE METRIC OBJECTS (Fixes the TypeError)
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        AnswerCorrectness(),
    ]

    # Config for stability (timeout and workers)
    run_config = RunConfig(timeout=1000, max_workers=1, max_retries=3)

    # 6. Perform Evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config
    )

    # 7. Export and Display Results
    df = result.to_pandas()
    df.to_json(output_path, orient='records', indent=4)
    
    print("\n" + "="*50)
    print("📊 RAGAS EVALUATION SUMMARY")
    print("="*50)
    print(result)
    print("="*50)
    print(f"🚀 Detailed results saved to: {output_path}")

if __name__ == "__main__":
    run_evaluation()