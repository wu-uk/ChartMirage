import os
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

# Try to find project root by looking for 'dataset' directory
if os.path.exists("dataset"):
    base_dir = "."
elif os.path.exists("../../dataset"):
    base_dir = "../.."
else:
    base_dir = ".."

import json
from defensive_rag_pipeline import DefensiveRAGPipeline

def verify_pipeline():
    pipeline = DefensiveRAGPipeline()
    
    # Load dataset to get captions
    dataset_path = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    entry = data[0]
    orig_caption = entry['original_caption']
    fake_caption = entry['fake_caption']
    orig_image = os.path.join(base_dir, "dataset", entry['image'])
    edited_image = os.path.join(base_dir, "dataset", entry['edited_image'])
    noise_image = os.path.join(base_dir, "dataset/images_noise_fake/gaussian_noise", os.path.basename(entry['edited_image']))

    tests = [
        ("Normal Case (Clean + Clean)", orig_image, orig_caption),
        ("Semantic Attack (Clean Image + Fake Text)", orig_image, fake_caption),
        ("Semantic Attack (Fake Image + Clean Text)", edited_image, orig_caption),
        ("Noise Attack (Noisy Image + Clean Text)", noise_image, orig_caption),
    ]

    print("\n" + "="*50)
    print("Starting Pipeline Verification")
    print("="*50)

    for name, img, txt in tests:
        print(f"\n>>> Running Test: {name}")
        print(f"Image: {img}")
        if not os.path.exists(img):
            print(f"ERROR: Image not found: {img}")
            continue
            
        is_safe, level, response = pipeline.process(img, txt)
        print(f"Result: {'SAFE' if is_safe else 'UNSAFE'}")
        print(f"Detected at: {level}")
        print(f"Response: {response}")

if __name__ == "__main__":
    verify_pipeline()
