import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "result"

def analyze_noise_impact():
    # 1. Load Baseline (Exp 3: Fake Image + Orig Caption)
    baseline_files = glob.glob(os.path.join(RESULT_DIR, "results_Exp_3*.json"))
    if not baseline_files:
        print("Warning: Baseline result (Exp 3) not found.")
        baseline_acc = 0
    else:
        with open(baseline_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
        total = 0
        correct = 0
        for item in data:
            total += 1
            if item.get("eval_result", {}).get("correct"):
                correct += 1
        baseline_acc = (correct / total * 100) if total > 0 else 0
        print(f"Baseline (No Noise): {baseline_acc:.2f}%")

    # 2. Load Noise Results
    noise_files = glob.glob(os.path.join(RESULT_DIR, "results_Noise_*.json"))
    results = []
    
    # Add Baseline
    results.append({"Noise Type": "None (Baseline)", "Accuracy": baseline_acc})

    for fpath in noise_files:
        noise_type = os.path.basename(fpath).replace("results_Noise_", "").replace(".json", "")
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        total = 0
        correct = 0
        for item in data:
            total += 1
            if item.get("eval_result", {}).get("correct"):
                correct += 1
        
        acc = (correct / total * 100) if total > 0 else 0
        results.append({
            "Noise Type": noise_type, 
            "Accuracy": acc,
            "Total": total,
            "Correct": correct
        })
        print(f"Noise ({noise_type}): {acc:.2f}% ({correct}/{total})")

    # 3. Create DataFrame and Plot
    df = pd.DataFrame(results)
    df = df.sort_values(by="Accuracy", ascending=False)
    
    print("\nSummary Table:")
    print(df[["Noise Type", "Accuracy", "Correct", "Total"]])
    
    csv_path = os.path.join(RESULT_DIR, "noise_analysis_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved summary to {csv_path}")

    # Plot
    try:
        plt.figure(figsize=(10, 6))
        bars = plt.barh(df["Noise Type"], df["Accuracy"], color='skyblue')
        plt.xlabel("Accuracy (%)")
        plt.title("Impact of Image Noise on QA Accuracy (Fake Image + Orig Caption)")
        plt.xlim(0, 100)
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center')
            
        plt.tight_layout()
        plot_path = os.path.join(RESULT_DIR, "noise_impact_plot.png")
        plt.savefig(plot_path)
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    analyze_noise_impact()
