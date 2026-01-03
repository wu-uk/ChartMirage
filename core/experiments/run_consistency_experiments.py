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

def run_experiment(exp_name, image_key, caption_key, dataset_path, args):
    print(f"\n{'='*60}")
    print(f"Starting Experiment: {exp_name}")
    print(f"Configuration: Image='{image_key}', Caption='{caption_key}'")
    print(f"{'='*60}")

    storage_dir = os.path.join(base_dir, f"storage/storage_consistency_{exp_name.lower().replace(' ', '_')}")

    
    # Initialize Pipeline
    pipeline = ConsistencyPipeline(
        dataset_path=dataset_path,
        storage_dir=storage_dir,
        eval_enabled=True,
        image_key=image_key,
        caption_key=caption_key
    )

    # Build Index
    # We force rebuild to ensure the index matches the current experiment's data configuration
    print(f"Building Index for {exp_name}...")
    pipeline.build_index(force_rebuild=True)

    # Load Queries
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if args.limit:
        print(f"Limit set to {args.limit} queries.")
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
            "image": entry.get(image_key), 
            "caption": entry.get(caption_key)
        })

    # Run Batch Query
    print(f"Running {len(query_items)} queries for {exp_name}...")
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
    
    print(f"\nResults for {exp_name}:")
    print(f"Total: {total}, Errors: {errors}")
    print(f"Refusals: {refusals} ({refusal_rate:.2f}%)")
    print(f"Accuracy: {correct} ({accuracy:.2f}%)")
    print(f"ASR: {attack_success} ({asr:.2f}%)")

    # Save Results
    output_file = os.path.join(base_dir, f"outputs/results/result_consistency_{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved results to {output_file}")
    
    return {
        "Experiment": exp_name,
        "Total": total,
        "Refusals": f"{refusal_rate:.2f}%",
        "Accuracy": f"{accuracy:.2f}%",
        "ASR": f"{asr:.2f}%",
        "File": output_file
    }

def main():
    parser = argparse.ArgumentParser(description="Run Consistency Pipeline on All Experiments")
    parser.add_argument("--workers", type=int, default=5, help="Number of threads")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries per experiment (for testing)")
    parser.add_argument("--exp", type=str, default="all", choices=["all", "1", "2", "3", "4"], help="Specific experiment to run")
    args = parser.parse_args()

    # Try to find project root by looking for 'dataset' directory
    if os.path.exists("dataset"):
        base_dir = "."
    elif os.path.exists("../../dataset"):
        base_dir = "../.."
    else:
        base_dir = ".."

    dataset_path = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    
    experiments = [

        ("Exp_1_Baseline", "image", "original_caption"),
        ("Exp_2_Text_Poison", "image", "fake_caption"),
        ("Exp_3_Image_Poison", "edited_image", "original_caption"),
        ("Exp_4_Dual_Poison", "edited_image", "fake_caption")
    ]

    summary = []

    for exp_name, img_key, cap_key in experiments:
        # Filter if specific experiment requested
        if args.exp != "all":
            if args.exp not in exp_name: # e.g. "1" in "Exp_1..."
                continue

        try:
            res = run_experiment(exp_name, img_key, cap_key, dataset_path, args)
            summary.append(res)
        except Exception as e:
            print(f"Error running {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print Final Summary
    print("\n" + "="*80)
    print(f"{'Experiment':<25} | {'Refusals':<10} | {'Accuracy':<10} | {'ASR':<10}")
    print("-" * 80)
    for s in summary:
        print(f"{s['Experiment']:<25} | {s['Refusals']:<10} | {s['Accuracy']:<10} | {s['ASR']:<10}")
    print("="*80)

if __name__ == "__main__":
    main()
