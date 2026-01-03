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

from core.experiments.consistency_pipeline import ConsistencyPipeline

def test_consistency():
    storage_dir = os.path.join(base_dir, "storage/storage_test_consistency")
    dataset_path = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    
    # Clean up
    if os.path.exists(storage_dir):
        shutil.rmtree(storage_dir)

    print("Initializing ConsistencyPipeline...")
    pipeline = ConsistencyPipeline(
        dataset_path=dataset_path,
        storage_dir=storage_dir,
        eval_enabled=False,
        image_key="image",
        caption_key="original_caption"
    )

    print("Building Index...")
    pipeline.build_index(force_rebuild=True)

    print("\nTest 1: Normal Query (Should Answer)")
    # Q: "What is shown in the chart?" (Generic)
    res1 = pipeline.process_single_query({"query": "What is shown in the chart?", "ground_truth": "Unknown"})
    print(f"Result 1: {res1.get('prediction')}")

    print("\nTest 2: Forced Inconsistency (Simulated)")
    # Since we are using "image" + "original_caption", they ARE consistent.
    # To test inconsistency, we'd need a query that exposes a conflict or use the "Fake Caption" experiment.
    # But for a quick test, we can check if the consistency check runs without crashing.
    
    # We can manually invoke check_consistency with conflicting info to verify logic
    # Mock nodes
    from llama_index.core.schema import TextNode, ImageNode
    text_node = TextNode(text="The trend is increasing.")
    # We need a valid image path for the check to work
    # Let's use an image from the dataset
    import json
    with open(dataset_path, "r") as f:
        data = json.load(f)
        first_img = data[0]["image"]
    
    # Resolve path
    abs_img_path = pipeline._resolve_image_path(first_img)
    image_node = ImageNode(image_path=abs_img_path)
    
    print("\nRunning Manual Consistency Check (Expect Inconsistent)...")
    consistent, reason = pipeline.check_consistency(
        "Is the trend increasing or decreasing?", 
        [text_node], 
        [image_node] # The image (likely complex chart) might not match "increasing" simply, or LLM might find it consistent.
        # Let's try a blatant lie in text
    )
    print(f"Consistent: {consistent}, Reason: {reason}")
    
    text_node_lie = TextNode(text="This is a picture of a cat.")
    print("\nRunning Manual Consistency Check (Expect Cat Lie)...")
    consistent_lie, reason_lie = pipeline.check_consistency(
        "What is in the image?", 
        [text_node_lie], 
        [image_node]
    )
    print(f"Consistent: {consistent_lie}, Reason: {reason_lie}")

if __name__ == "__main__":
    test_consistency()
