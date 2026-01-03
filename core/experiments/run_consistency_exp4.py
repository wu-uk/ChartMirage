import os
import sys
import json
import argparse
from datetime import datetime

# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Try to find project root by looking for 'dataset' directory
if os.path.exists("dataset"):
    base_dir = "."
elif os.path.exists("../../dataset"):
    base_dir = "../.."
else:
    base_dir = ".."

from core.experiments.consistency_pipeline import ConsistencyPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Experiment 4: Aligned Fake Image + Fake Caption")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries for testing")
    args = parser.parse_args()

    dataset_path = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    storage_dir = os.path.join(base_dir, "storage/storage_consistency_exp4")
    image_key = "edited_image"
    caption_key = "fake_caption"

    
    # Initialize Pipeline
    print("Initializing ConsistencyPipeline for Exp 4...")
    pipeline = ConsistencyPipeline(
        dataset_path=dataset_path,
        storage_dir=storage_dir,
        eval_enabled=True,
        image_key=image_key,
        caption_key=caption_key
    )

    # Build Index (Force rebuild to ensure we use the fake data)
    print("Building Index...")
    pipeline.build_index(force_rebuild=True)

    # Load Queries
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if args.limit:
        data = data[:args.limit]
    
    # Prepare items
    query_items = []
    for entry in data:
        if not entry.get("query"):
            continue
        query_items.append({
            "query": entry["query"],
            "ground_truth": entry.get("answer"),
            "fake_answer": entry.get("fake_answer"),
            "image": entry.get(image_key), # Just for reference if needed
            "caption": entry.get(caption_key)
        })

    # Run Batch Query
    print(f"Running {len(query_items)} queries...")
    results = pipeline.batch_query(query_items, max_workers=args.workers)

    # Calculate Statistics
    total = len(results)
    refusals = 0
    correct = 0
    attack_success = 0
    errors = 0

    for res in results:
        if res.get("error"):
            errors += 1
            continue
            
        if res.get("is_refusal"):
            refusals += 1
        else:
            # Accuracy
            if res.get("eval_result", {}).get("correct"):
                correct += 1
            
            # ASR
            if res.get("asr_result", {}).get("match"):
                attack_success += 1

    # Statistics
    refusal_rate = (refusals / total) * 100 if total > 0 else 0
    accuracy = (correct / total) * 100 if total > 0 else 0
    asr = (attack_success / total) * 100 if total > 0 else 0
    
    # For ASR, we usually care about Success / (Total - Refusals) if we consider refusals as "Defense Success"
    # But standard ASR is Success / Total Attempts.
    
    print("\n" + "="*50)
    print(f"Consistency Pipeline Evaluation Results (Exp 4)")
    print("="*50)
    print(f"Total Queries: {total}")
    print(f"Errors: {errors}")
    print(f"Refusals: {refusals} ({refusal_rate:.2f}%)")
    print(f"Correct Answers: {correct} ({accuracy:.2f}%)")
    print(f"Attack Successes: {attack_success} ({asr:.2f}%)")
    print("="*50)

    # Save Results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(base_dir, f"outputs/results/result_consistency_exp4_{timestamp}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
