import json
import os

def check_paths():
    dataset_path = os.path.join("dataset", "final_qa_merged_unified.json")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join("..", "dataset", "final_qa_merged_unified.json")
    
    print(f"Loading dataset from: {dataset_path}")
    if not os.path.exists(dataset_path):
        print("Dataset file not found!")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total entries: {len(data)}")
    
    found_count = 0
    missing_count = 0
    
    # Check first 50 entries
    for i, entry in enumerate(data[:50]):
        img_rel_path = entry["edited_image"]
        
        # Logic copied from exp_edited_image.py
        img_rel_path_norm = img_rel_path.replace("/", os.sep)
        img_abs_path = None
        
        candidates = [
            os.path.join("dataset", img_rel_path_norm),
            os.path.join("..", "dataset", img_rel_path_norm)
        ]
        
        parts = img_rel_path_norm.split(os.sep)
        if len(parts) > 1:
            subdir = parts[0]
            rest = os.path.join(*parts[1:])
            candidates.append(os.path.join("dataset", subdir, subdir, rest))
            candidates.append(os.path.join("..", "dataset", subdir, subdir, rest))
        
        for p in candidates:
            if os.path.exists(p):
                img_abs_path = os.path.abspath(p)
                break
        
        if img_abs_path:
            found_count += 1
            print(f"[OK] Found: {img_abs_path}")
        else:
            missing_count += 1
            print(f"[FAIL] Missing: {img_rel_path}")

    print(f"\nSummary (first 50): Found={found_count}, Missing={missing_count}")

if __name__ == "__main__":
    check_paths()
