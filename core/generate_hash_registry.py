import json
import hashlib
import os

def generate_hash_registry(json_path, image_root):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    registry = {}
    for entry in data:
        image_rel_path = entry.get('image')
        text_content = entry.get('original_caption')
        
        if image_rel_path and text_content:
            image_path = os.path.join(image_root, image_rel_path)
            if os.path.exists(image_path):
                # Calculate hash for (Image Bytes + Text)
                with open(image_path, 'rb') as f_img:
                    img_bytes = f_img.read()
                
                hasher = hashlib.sha256()
                hasher.update(img_bytes)
                hasher.update(text_content.encode('utf-8'))
                combined_hash = hasher.hexdigest()
                
                # Store it
                registry[combined_hash] = True
                
    return registry

if __name__ == "__main__":
    # Try to find project root by looking for 'dataset' directory
    if os.path.exists("dataset"):
        base_dir = "."
    elif os.path.exists("../../dataset"):
        base_dir = "../.."
    else:
        base_dir = ".."

    dataset_path = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    image_root = os.path.join(base_dir, "dataset") # images_merged is inside dataset/
    output_path = os.path.join(base_dir, "dataset/hash_registry.json")
    
    print("Generating hash registry...")
    registry = generate_hash_registry(dataset_path, image_root)
    with open(output_path, 'w') as f:
        json.dump(registry, f)
    print(f"Registry saved to {output_path} with {len(registry)} entries.")
