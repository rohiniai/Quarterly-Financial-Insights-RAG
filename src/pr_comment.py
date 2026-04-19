import json
import os
import pandas as pd

def format_results():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_dir, 'data', 'eval_ragas_results.json')
    
    if not os.path.exists(results_path):
        return "❌ Error: Evaluation results file not found."

    with open(results_path, 'r') as f:
        data = json.load(f)
    
    if not data:
        return "⚠️ Evaluation completed, but no data was returned."

    df = pd.DataFrame(data)
    
    # Calculate mean scores
    metrics = [
        'faithfulness', 
        'answer_relevancy',  
        'context_recall', 
        'answer_correctness'
    ]
    
    # Check if metrics exist in the data
    existing_metrics = [m for m in metrics if m in df.columns]
    
    if not existing_metrics:
        return "⚠️ No Ragas metrics found in the results file."
    
    summary = "## 📊 RAG Evaluation Summary\n\n"
    summary += "| Metric | Mean Score |\n"
    summary += "| --- | --- |\n"
    
    for m in existing_metrics:
        mean_val = df[m].mean()
        status = "✅" if mean_val >= 0.8 else "⚠️" if mean_val >= 0.6 else "❌"
        summary += f"| {m.replace('_', ' ').title()} | {mean_val:.4f} {status} |\n"
    
    summary += "\n### 🔍 Detailed Results (Per Question)\n\n"
    summary += "| Question | Faithfulness | Relevancy | Correctness |\n"
    summary += "| --- | --- | --- | --- |\n"
    
    # Limit to top 5 for the comment to avoid bloat
    for _, row in df.head(5).iterrows():
        summary += f"| {row['question'][:50]}... | {row.get('faithfulness', 'N/A'):.2f} | {row.get('answer_relevancy', 'N/A'):.2f} | {row.get('answer_correctness', 'N/A'):.2f} |\n"
    
    summary += f"\n*Full results saved in `data/eval_ragas_results.json`*"
    
    return summary

if __name__ == "__main__":
    print(format_results())
