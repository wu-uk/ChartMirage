import os
import sys
import json
import argparse
import random
import copy
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

def create_mixed_dataset(original_path, ratio, output_path, seed=42):
    """
    Create a dataset with mixed real and fake images.
    ratio: float, percentage of fake images (0.0 to 1.0)
    """
    with open(original_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    random.seed(seed)
    # Shuffle indices to pick which ones to poison
    indices = list(range(len(data)))
    random.shuffle(indices)
    
    num_poison = int(len(data) * ratio)
    poison_indices = set(indices[:num_poison])
    
    mixed_data = []
    poisoned_count = 0
    
    for i, entry in enumerate(data):
        new_entry = entry.copy()
        
        # Determine if this entry should be poisoned (fake image)
        if i in poison_indices:
            # Use fake image if available
            if entry.get("edited_image"):
                new_entry["image_to_use"] = entry["edited_image"]
                new_entry["is_poisoned"] = True
                poisoned_count += 1
            else:
                # Fallback if no fake image (shouldn't happen in this dataset usually)
                new_entry["image_to_use"] = entry["image"]
                new_entry["is_poisoned"] = False
        else:
            # Use real image
            new_entry["image_to_use"] = entry["image"]
            new_entry["is_poisoned"] = False
            
        # Caption is always original
        new_entry["caption_to_use"] = entry["original_caption"]
        
        mixed_data.append(new_entry)
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mixed_data, f, indent=2, ensure_ascii=False)
        
    return poisoned_count, len(mixed_data)

def run_ablation(ratio, args):
    print(f"\n{'='*60}")
    print(f"Starting Ablation: Mixed Real/Fake (Ratio={ratio})")
    print(f"{'='*60}")

    dataset_source = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    temp_dataset = os.path.join(base_dir, f"dataset/temp_ablation_{int(ratio*100)}.json")
    
    # 1. Create Mixed Dataset
    p_count, total = create_mixed_dataset(dataset_source, ratio, temp_dataset)
    print(f"Created mixed dataset: {p_count}/{total} poisoned items ({p_count/total*100:.2f}%)")

    # 2. Initialize Pipeline
    # Storage needs to be separate for each ratio to avoid index conflicts
    storage_dir = os.path.join(base_dir, f"storage/storage_ablation_{int(ratio*100)}")

    
    # We use "image_to_use" and "caption_to_use" keys we created
    pipeline = ConsistencyPipeline(
        dataset_path=temp_dataset,
        storage_dir=storage_dir,
        eval_enabled=True,
        image_key="image_to_use",
        caption_key="caption_to_use"
    )

    # 3. Build Index
    print(f"Building Index for {ratio*100}% ablation...")
    pipeline.build_index(force_rebuild=True)

    # 4. Load Data for Querying
    with open(temp_dataset, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if args.limit:
        print(f"Limit set to {args.limit} queries.")
        data = data[:args.limit]
    
    query_items = []
    for entry in data:
        if not entry.get("query"):
            continue
        query_items.append({
            "query": entry["query"],
            "ground_truth": entry.get("answer"),
            "fake_answer": entry.get("fake_answer"),
            "is_poisoned": entry.get("is_poisoned") # Track if this specific item was poisoned
        })

    # 5. Run Batch Query
    print(f"Running {len(query_items)} queries...")
    results = pipeline.batch_query(query_items, max_workers=args.workers)

    # 6. Analyze Results
    stats = {
        "total": len(results),
        "poisoned_subset": {"total": 0, "refusals": 0, "correct": 0, "asr": 0},
        "clean_subset": {"total": 0, "refusals": 0, "correct": 0, "asr": 0}
    }

    for res in results:
        # Determine subset based on input item info (we passed it through? No, batch_query returns result dicts)
        # We need to map back or store metadata in result. 
        # ConsistencyPipeline.process_single_query returns a new dict.
        # But we can assume order is preserved or use query string matching if unique.
        # Actually, let's look at how batch_query works. It returns a list.
        # However, ThreadPoolExecutor might scramble order? 
        # Wait, as_completed yields futures as they complete. Order is NOT preserved.
        # We need to match results back to inputs.
        # The pipeline implementation:
        # future_to_item = {executor.submit(...): item for item in query_items}
        # ...
        # item = future_to_item[future] (in exception block only in my memory? Let's check)
        # Ah, in success block: res = future.result(). It doesn't attach the original item metadata unless process_single_query does.
        # process_single_query returns {query, prediction, ...}
        # It does NOT return our custom "is_poisoned" flag unless we modify the pipeline.
        
        # Workaround: The result contains "query". We can try to match. 
        # But duplicate queries might exist.
        # Better: We modify `query_items` passed to `process_single_query` to include `is_poisoned`, 
        # and `process_single_query` returns whatever it gets?
        # `process_single_query` takes `item`. It extracts `query`, `ground_truth`. 
        # It returns a dict constructed from scratch.
        
        # Let's rely on matching by query for now (assuming low collision) or just re-calculate stats broadly.
        # Actually, for this specific analysis, distinguishing Clean vs Poisoned performance is CRITICAL.
        
        # I will match by query text.
        pass

    # Re-matching logic
    # Create a map of query -> is_poisoned (Warning: duplicate queries will be ambiguous)
    # Let's check if there are duplicate queries.
    query_map = {} # query -> [is_poisoned, ...]
    for item in query_items:
        q = item["query"]
        if q not in query_map:
            query_map[q] = []
        query_map[q].append(item["is_poisoned"])
    
    # Consumption tracking for duplicates
    query_consumption = {k: 0 for k in query_map}

    for res in results:
        q = res.get("query")
        if not q or q not in query_map:
            continue
            
        # Get is_poisoned status
        idx = query_consumption[q]
        if idx < len(query_map[q]):
            is_poisoned = query_map[q][idx]
            query_consumption[q] += 1
        else:
            is_poisoned = query_map[q][0] # Fallback
            
        subset = stats["poisoned_subset"] if is_poisoned else stats["clean_subset"]
        subset["total"] += 1
        
        if res.get("is_refusal"):
            subset["refusals"] += 1
        else:
            if res.get("eval_result", {}).get("correct"):
                subset["correct"] += 1
            if res.get("asr_result", {}).get("match"):
                subset["asr"] += 1

    # Print Summary
    print(f"\nResults for {ratio*100}% Poisoning:")
    
    def print_sub(name, s):
        t = s["total"]
        if t == 0:
            print(f"  {name}: No data")
            return
        r_rate = s["refusals"] / t * 100
        acc = s["correct"] / t * 100
        asr = s["asr"] / t * 100
        print(f"  {name} (N={t}): Refusal={r_rate:.1f}%, Acc={acc:.1f}%, ASR={asr:.1f}%")
        
    print_sub("Clean Subset", stats["clean_subset"])
    print_sub("Poisoned Subset", stats["poisoned_subset"])
    
    # Overall
    total = stats["total"]
    all_ref = stats["clean_subset"]["refusals"] + stats["poisoned_subset"]["refusals"]
    all_acc = stats["clean_subset"]["correct"] + stats["poisoned_subset"]["correct"]
    all_asr = stats["clean_subset"]["asr"] + stats["poisoned_subset"]["asr"]
    
    print(f"  Overall (N={total}): Refusal={all_ref/total*100:.1f}%, Acc={all_acc/total*100:.1f}%, ASR={all_asr/total*100:.1f}%")

    # Save
    out_file = os.path.join(base_dir, f"outputs/results/result_ablation_{int(ratio*100)}.json")
    final_data = {
        "ratio": ratio,
        "stats": stats,
        "results": results
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
        
    return {
        "Ratio": f"{ratio*100}%",
        "Overall Refusal": f"{all_ref/total*100:.1f}%",
        "Overall Acc": f"{all_acc/total*100:.1f}%",
        "Overall ASR": f"{all_asr/total*100:.1f}%",
        "Clean Refusal": f"{stats['clean_subset']['refusals']/stats['clean_subset']['total']*100:.1f}%" if stats['clean_subset']['total'] else "N/A",
        "Poisoned Refusal": f"{stats['poisoned_subset']['refusals']/stats['poisoned_subset']['total']*100:.1f}%" if stats['poisoned_subset']['total'] else "N/A"
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    ratios = [0.25, 0.50, 0.75]
    summary_table = []
    
    for r in ratios:
        try:
            res = run_ablation(r, args)
            summary_table.append(res)
        except Exception as e:
            print(f"Error at {r}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "="*100)
    print(f"{'Ratio':<10} | {'Ov. Refusal':<12} | {'Ov. Acc':<10} | {'Ov. ASR':<10} | {'Clean Ref.':<12} | {'Poison Ref.':<12}")
    print("-" * 100)
    for row in summary_table:
        print(f"{row['Ratio']:<10} | {row['Overall Refusal']:<12} | {row['Overall Acc']:<10} | {row['Overall ASR']:<10} | {row['Clean Refusal']:<12} | {row['Poisoned Refusal']:<12}")
    print("="*100)

if __name__ == "__main__":
    main()
