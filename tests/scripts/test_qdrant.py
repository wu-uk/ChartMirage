import os
import sys
import shutil

# Add project root
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

from core.experiments.qdrant_pipeline import QdrantPipeline

def test_qdrant():
    storage_dir = os.path.join(base_dir, "storage/storage_test_qdrant")
    dataset_path = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    
    # Clean up previous test run
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

    print("Initializing QdrantPipeline...")
    pipeline = QdrantPipeline(
        dataset_path=dataset_path,
        storage_dir=storage_dir,
        eval_enabled=False, # Disable eval to save time/tokens for this test
        image_key="image",
        caption_key="original_caption"
    )

    print("Building Index...")
    pipeline.build_index(force_rebuild=True)

    print("Running a test query...")
    # Manual query
    response, _ = pipeline.query("What is shown in the chart?", ground_truth=None)
    print(f"Response: {response}")

    # Batch query test
    query_items = [
        {"query": "What is the title of the chart?", "ground_truth": "Unknown", "id": 1},
        {"query": "Describe the trend.", "ground_truth": "Unknown", "id": 2}
    ]
    print("Running batch query...")
    results = pipeline.batch_query(query_items, max_workers=2)
    print("Batch Results:", results)

if __name__ == "__main__":
    test_qdrant()
