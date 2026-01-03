import json
import pandas as pd
import os

# Try to find project root by looking for 'dataset' directory
if os.path.exists("dataset"):
    base_dir = "."
elif os.path.exists("../../dataset"):
    base_dir = "../.."
else:
    base_dir = ".."

def analyze_partial():
    results = []
    results_path = os.path.join(base_dir, "outputs/results/baseline_b_partial.jsonl")
    if not os.path.exists(results_path):
        # Fallback to current directory for backward compatibility
        results_path = "baseline_b_partial.jsonl"
        
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        for line in f:
            results.append(json.loads(line))
    
    df = pd.DataFrame(results)
    
    safe_mask = df["True Label"] == "SAFE"
    unsafe_mask = df["True Label"] == "UNSAFE"
    
    fp = ((df["B Pred"] == "UNSAFE") & safe_mask).sum()
    fn = ((df["B Pred"] == "SAFE") & unsafe_mask).sum()
    
    total_safe = safe_mask.sum()
    total_unsafe = unsafe_mask.sum()
    
    fpr = fp / total_safe if total_safe > 0 else 0
    fnr = fn / total_unsafe if total_unsafe > 0 else 0
    
    print("\n" + "="*50)
    print("PARTIAL BASELINE B RESULTS (N=197)")
    print("="*50)
    print(f"Total Samples: {len(df)}")
    print(f"SAFE Samples: {total_safe}, UNSAFE Samples: {total_unsafe}")
    print(f"False Positives: {fp}, False Negatives: {fn}")
    print(f"False Positive Rate (FPR): {fpr:.2%}")
    print(f"False Negative Rate (FNR): {fnr:.2%}")
    print("="*50)

if __name__ == "__main__":
    analyze_partial()
