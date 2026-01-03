import os
import json
import sys
from llama_index.core import Settings

# Add project root to sys.path
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

MAX_WORKERS = 20
RESULT_DIR = os.path.join(base_dir, "outputs/results")

def run_experiment(name, image_key, caption_key, storage_dir, limit=5):
    print(f"\n{'='*50}")
    print(f"Running Experiment (Qdrant): {name}")
    print(f"Configuration: Image={image_key}, Caption={caption_key}")
    print(f"{'='*50}")

    dataset_path = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    
    # Initialize Pipeline
    pipeline = ConsistencyPipeline(  
        dataset_path=dataset_path,
        storage_dir=storage_dir,
        eval_enabled=True,
        image_key=image_key,
        caption_key=caption_key
    )
    
    # Build Index
    try:
        # Check if Qdrant DB exists within storage_dir
        # Structure: storage_dir/qdrant_db
        # Also storage_dir/docstore.json etc.
        should_rebuild = True
        if os.path.exists(storage_dir) and os.path.exists(os.path.join(storage_dir, "docstore.json")):
             should_rebuild = False
            
        pipeline.build_index(force_rebuild=should_rebuild)
    except Exception as e:
        print(f"Index build failed: {e}")
        return

    # Load Data for Querying
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Run Queries
    print(f"Preparing queries (limit={limit})...")
    query_items = []
    count = 0
    for entry in data:
        if count >= limit:
            break
            
        query_str = entry.get("query")
        answer = entry.get("answer")
        fake_answer = entry.get("fake_answer")
        
        # Skip if answers are identical (less interesting for this test)
        if answer == fake_answer:
            continue
            
        # Prepare item for batch processing
        item = {
            "query": query_str,
            "ground_truth": answer,
            "fake_answer": fake_answer, # Store for post-processing/comparison
            "id": count
        }
        query_items.append(item)
        count += 1
    
    # Execute Batch
    results = pipeline.batch_query(query_items, max_workers=MAX_WORKERS)
    
    # Save Results
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    output_file = f"{RESULT_DIR}/results_{name.replace(' ', '_').replace(':', '').replace('(', '').replace(')', '')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")
    
    # Print Summary of first few
    for res in results[:3]:
        print(f"\nQ: {res.get('query')}")
        print(f"Pred: {res.get('prediction')}")
        eval_res = res.get("eval_result")
        if eval_res:
            print(f"Eval: {'✅' if eval_res.get('correct') else '❌'} - {eval_res.get('reason')}")

if __name__ == "__main__":
    # Define 4 Experiments
    experiments = [
        {
            "name": "Exp 1: Baseline (Orig Image + Orig Caption)",
            "image_key": "image",
            "caption_key": "original_caption",
            "storage_dir": "storage/storage_qdrant_exp_OO"
        },
        {
            "name": "Exp 2: Caption Conflict (Orig Image + Fake Caption)",
            "image_key": "image",
            "caption_key": "fake_caption",
            "storage_dir": "storage/storage_qdrant_exp_OF"
        },
        {
            "name": "Exp 3: Image Conflict (Fake Image + Orig Caption)",
            "image_key": "edited_image",
            "caption_key": "original_caption",
            "storage_dir": "storage/storage_qdrant_exp_FO"
        },
        {
            "name": "Exp 4: Aligned Fake (Fake Image + Fake Caption)",
            "image_key": "edited_image",
            "caption_key": "fake_caption",
            "storage_dir": "storage/storage_qdrant_exp_FF"
        }
    ]

    for exp in experiments:
        storage_dir = os.path.join(base_dir, exp["storage_dir"])
        run_experiment(
            exp["name"], 
            exp["image_key"], 
            exp["caption_key"], 
            storage_dir,
            limit=10000  # Set a large limit to process all data
        )
