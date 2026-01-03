import json
import os
import glob
import pandas as pd

def calculate_accuracy(file_path):
    if not os.path.exists(file_path):
        return None
    
    total = 0
    correct = 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if "evaluation" in entry:
                        total += 1
                        if entry["evaluation"].get("correct"):
                            correct += 1
                except:
                    pass
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
        
    if total == 0:
        return 0.0
    
    return (correct / total) * 100

def main():
    # Try to find project root by looking for 'outputs' directory
    if os.path.exists("outputs"):
        base_dir = "."
    elif os.path.exists("../../outputs"):
        base_dir = "../.."
    else:
        base_dir = ".."
        
    result_dir = os.path.join(base_dir, "outputs/results")
    files = sorted(glob.glob(os.path.join(result_dir, "*_evaluated.jsonl")))

    
    results = []
    
    print(f"{'Experiment':<60} | {'Accuracy':<10} | {'ASR':<10}")
    print("-" * 90)
    
    for file_path in files:
        if "_asr_evaluated.jsonl" in file_path:
            continue
            
        acc = calculate_accuracy(file_path)
        
        # Try to find corresponding ASR file
        asr_file = file_path.replace("_evaluated.jsonl", "_asr_evaluated.jsonl")
        asr_val = "N/A"
        if os.path.exists(asr_file):
            asr_score = calculate_asr(asr_file)
            if asr_score is not None:
                asr_val = f"{asr_score:.2f}%"
        
        if acc is not None:
            name = os.path.basename(file_path).replace("results_", "").replace("_evaluated.jsonl", "")
            print(f"{name:<60} | {acc:.2f}%     | {asr_val:<10}")
            results.append({"Experiment": name, "Accuracy": acc, "ASR": asr_val})

def calculate_asr(file_path):
    if not os.path.exists(file_path):
        return None
    
    total = 0
    success = 0
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if "asr_evaluation" in entry:
                        total += 1
                        if entry["asr_evaluation"].get("match"):
                            success += 1
                except:
                    pass
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
        
    if total == 0:
        return 0.0
    
    return (success / total) * 100

            
    # Save summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(result_dir, "evaluation_summary.csv"), index=False)
    print(f"\nSummary saved to {os.path.join(result_dir, 'evaluation_summary.csv')}")

if __name__ == "__main__":
    main()
