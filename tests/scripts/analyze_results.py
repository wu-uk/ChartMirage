import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def analyze_results(result_dir="result_qdrant"):
    # Find all result JSON files
    json_files = glob.glob(os.path.join(result_dir, "results_Exp_*.json"))
    
    if not json_files:
        print(f"No result files found in {result_dir}")
        return

    summary = []
    
    print(f"{'Experiment Name':<60} | {'Total':<6} | {'Correct':<8} | {'Accuracy':<8}")
    print("-" * 100)

    for json_file in sorted(json_files):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
            
        exp_name = os.path.basename(json_file).replace("results_", "").replace(".json", "").replace("_", " ")
        
        total = 0
        correct = 0
        error_count = 0
        
        for item in data:
            total += 1
            if item.get("error"):
                error_count += 1
                continue
                
            eval_res = item.get("eval_result")
            if eval_res and eval_res.get("correct"):
                correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        summary.append({
            "Experiment": exp_name,
            "Total": total,
            "Correct": correct,
            "Errors": error_count,
            "Accuracy": accuracy
        })
        
        print(f"{exp_name:<60} | {total:<6} | {correct:<8} | {accuracy:.2f}%")

    # Create Summary DataFrame
    df = pd.DataFrame(summary)
    
    # Save Summary to CSV
    csv_path = os.path.join(result_dir, "analysis_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")
    
    # Optional: Plot
    try:
        plt.figure(figsize=(12, 6))
        # sns.barplot(data=df, y="Experiment", x="Accuracy", palette="viridis")
        # Fallback to matplotlib barh
        plt.barh(df["Experiment"], df["Accuracy"], color='skyblue')
        plt.title("Accuracy by Experiment Configuration")
        plt.xlabel("Accuracy (%)")
        plt.tight_layout()
        plot_path = os.path.join(result_dir, "accuracy_plot.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    analyze_results()
