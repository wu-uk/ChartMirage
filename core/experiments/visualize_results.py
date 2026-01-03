import json
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd
import numpy as np

# Set style manually
plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']

def calculate_stats(results):
    total = len(results)
    if total == 0:
        return 0, 0, 0
    
    refusals = sum(1 for r in results if r.get("is_refusal", False))
    
    correct = 0
    for r in results:
        if "eval_result" in r and isinstance(r["eval_result"], dict):
            if r["eval_result"].get("correct", False):
                correct += 1
        elif r.get("correct", False):
            correct += 1
            
    asr = 0
    for r in results:
        if "asr_result" in r and isinstance(r["asr_result"], dict):
            if r["asr_result"].get("match", False):
                asr += 1
        elif r.get("asr_match", False):
            asr += 1
            
    return (refusals / total) * 100, (correct / total) * 100, (asr / total) * 100

def load_exp_results(file_pattern, base_dir="."):
    data = []
    full_pattern = os.path.join(base_dir, "outputs/results", file_pattern)
    files = glob.glob(full_pattern)
    print(f"Looking for files with pattern: {full_pattern}")
    print(f"Found {len(files)} files.")
    
    grouped_files = {}
    for f in files:
        base_name = os.path.basename(f)
        if "Exp_1" in base_name: key = "Exp 1: Baseline"
        elif "Exp_2" in base_name: key = "Exp 2: Text Poison"
        elif "Exp_3" in base_name: key = "Exp 3: Image Poison"
        elif "Exp_4" in base_name: key = "Exp 4: Dual Poison"
        else: continue
        
        if key not in grouped_files:
            grouped_files[key] = []
        grouped_files[key].append(f)
    
    for key, file_list in grouped_files.items():
        latest_file = sorted(file_list)[-1]
        print(f"Loading {key} from {latest_file}...")
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                
            refusal, acc, asr = calculate_stats(results)
            data.append({
                "Experiment": key,
                "Refusal Rate": refusal,
                "Accuracy": acc,
                "ASR": asr
            })
        except Exception as e:
            print(f"Error loading {latest_file}: {e}")
            
    return pd.DataFrame(data)

def load_ablation_results(base_dir="."):
    data = []
    results_dir = os.path.join(base_dir, "outputs/results")
    # 0% comes from Baseline (Exp 1)
    baseline_files = glob.glob(os.path.join(results_dir, "result_consistency_Exp_1_Baseline_*.json"))
    if baseline_files:
        latest_baseline = sorted(baseline_files)[-1]
        try:
            with open(latest_baseline, 'r') as f:
                res = json.load(f)
            ref, acc, asr = calculate_stats(res)
            data.append({"Ratio": 0, "Refusal Rate": ref, "Accuracy": acc, "ASR": asr, "Type": "Overall"})
        except:
            pass

    for ratio in [25, 50, 75]:
        fname = os.path.join(results_dir, f"result_ablation_simulated_{ratio}.json")
        if os.path.exists(fname):
            print(f"Loading Ablation {ratio}% from {fname}...")
            try:
                with open(fname, 'r') as f:
                    content = json.load(f)
                
                if "stats" in content:
                    stats = content["stats"]

                    total = stats["total"]
                    clean = stats["clean"]
                    poison = stats["poisoned"]
                    
                    ov_ref = (clean["refusals"] + poison["refusals"]) / total * 100
                    ov_acc = (clean["correct"] + poison["correct"]) / total * 100
                    ov_asr = (clean["asr"] + poison["asr"]) / total * 100
                    
                    data.append({"Ratio": ratio, "Refusal Rate": ov_ref, "Accuracy": ov_acc, "ASR": ov_asr, "Type": "Overall"})
                    data.append({"Ratio": ratio, "Refusal Rate": (clean["refusals"]/clean["total"]*100), 
                                 "Accuracy": (clean["correct"]/clean["total"]*100), 
                                 "ASR": (clean["asr"]/clean["total"]*100), "Type": "Clean Subset"})
                    data.append({"Ratio": ratio, "Refusal Rate": (poison["refusals"]/poison["total"]*100), 
                                 "Accuracy": (poison["correct"]/poison["total"]*100), 
                                 "ASR": (poison["asr"]/poison["total"]*100), "Type": "Poisoned Subset"})
            except Exception as e:
                print(f"Error loading {fname}: {e}")

    return pd.DataFrame(data)

def plot_main_experiments(df):
    if df.empty:
        print("No main experiment data found.")
        return

    # Prepare data for plotting
    experiments = df["Experiment"].unique()
    x = np.arange(len(experiments))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    rects1 = ax.bar(x - width, df["Refusal Rate"], width, label='Refusal Rate', color='#4c72b0')
    rects2 = ax.bar(x, df["Accuracy"], width, label='Accuracy', color='#55a868')
    rects3 = ax.bar(x + width, df["ASR"], width, label='ASR', color='#c44e52')
    
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Performance of Consistency Checking Across Attacks')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=15)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig("plot_main_experiments.png", dpi=300)
    print("Saved plot_main_experiments.png")

def plot_ablation_study(df):
    if df.empty:
        print("No ablation data found.")
        return

    df_overall = df[df["Type"] == "Overall"]
    
    plt.figure(figsize=(10, 6))
    plt.plot(df_overall["Ratio"], df_overall["Refusal Rate"], marker='o', label="Refusal Rate", linewidth=2.5, color='#4c72b0')
    plt.plot(df_overall["Ratio"], df_overall["Accuracy"], marker='s', label="Accuracy", linewidth=2.5, color='#55a868')
    plt.plot(df_overall["Ratio"], df_overall["ASR"], marker='^', label="ASR (Attack Success)", linewidth=2.5, color='#c44e52')
    
    plt.title("Impact of Poison Injection Ratio on Pipeline Performance")
    plt.xlabel("Poison Injection Ratio (%)")
    plt.ylabel("Percentage (%)")
    plt.xticks([0, 25, 50, 75])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(plot_dir, "plot_ablation_trend.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    
    # Subsets plot using grouped bar chart
    df_subsets = df[df["Type"] != "Overall"]
    if not df_subsets.empty:
        ratios = sorted(df_subsets["Ratio"].unique())
        metrics = ["Refusal Rate", "Accuracy", "ASR"]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            x = np.arange(len(ratios))
            width = 0.35
            
            clean_vals = df_subsets[df_subsets["Type"] == "Clean Subset"][metric].values
            poison_vals = df_subsets[df_subsets["Type"] == "Poisoned Subset"][metric].values
            
            # Align lengths if missing data
            if len(clean_vals) != len(ratios):
                # Simple fallback or skip
                continue
                
            ax.bar(x - width/2, clean_vals, width, label='Clean Subset', color='#8172b3')
            ax.bar(x + width/2, poison_vals, width, label='Poisoned Subset', color='#ccb974')
            
            ax.set_title(metric)
            ax.set_xticks(x)
            ax.set_xticklabels(ratios)
            ax.set_xlabel("Ratio (%)")
            if i == 0:
                ax.set_ylabel("Percentage (%)")
                ax.legend()
                
        plt.suptitle("Clean vs. Poisoned Subset Performance at Different Ratios", fontsize=16)
        plt.tight_layout()
        save_path = os.path.join(plot_dir, "plot_ablation_subsets.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved {save_path}")

def load_no_defense_results(file_pattern="result_no_defense_*.json", base_dir="."):
    data = []
    full_pattern = os.path.join(base_dir, "outputs/results", file_pattern)
    files = glob.glob(full_pattern)
    print(f"Looking for No Defense files with pattern: {full_pattern}")
    print(f"Found {len(files)} files.")
    
    grouped_files = {}
    for f in files:
        base_name = os.path.basename(f)
        if "Exp_1" in base_name: key = "Exp 1: Baseline"
        elif "Exp_2" in base_name: key = "Exp 2: Text Poison"
        elif "Exp_3" in base_name: key = "Exp 3: Image Poison"
        elif "Exp_4" in base_name: key = "Exp 4: Dual Poison"
        else: continue
        
        if key not in grouped_files:
            grouped_files[key] = []
        grouped_files[key].append(f)
    
    for key, file_list in grouped_files.items():
        latest_file = sorted(file_list)[-1]
        print(f"Loading {key} (No Defense) from {latest_file}...")
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                
            refusal, acc, asr = calculate_stats(results)
            data.append({
                "Experiment": key,
                "Refusal Rate": refusal,
                "Accuracy": acc,
                "ASR": asr,
                "Defense": "Off"
            })
        except Exception as e:
            print(f"Error loading {latest_file}: {e}")
            
    return pd.DataFrame(data)

def plot_defense_comparison(df_defense, df_no_defense, plot_dir="."):
    if df_defense.empty or df_no_defense.empty:
        print("Missing data for comparison.")
        return

    # Add Defense column to df_defense if missing
    if "Defense" not in df_defense.columns:
        df_defense["Defense"] = "On"
        
    # Combine
    df_combined = pd.concat([df_defense, df_no_defense], ignore_index=True)
    
    # Metrics to plot: Accuracy, ASR
    # Refusal is interesting too
    
    metrics = ["Accuracy", "ASR", "Refusal Rate"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    experiments = sorted(df_combined["Experiment"].unique())
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Prepare data for this metric
        # Pivot: Index=Experiment, Columns=Defense, Values=Metric
        df_pivot = df_combined.pivot(index="Experiment", columns="Defense", values=metric)
        
        # Reorder index if needed
        df_pivot = df_pivot.reindex(experiments)
        
        # Plot
        df_pivot.plot(kind="bar", ax=ax, color={"On": "#55a868", "Off": "#c44e52"}, width=0.7)
        
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("Percentage (%)")
        ax.set_xticklabels(experiments, rotation=20, ha="right")
        ax.legend(title="Consistency Check")
        
        # Add values
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3, fontsize=9)

    plt.suptitle("Impact of Consistency Checking Defense on RAG Performance", fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(plot_dir, "plot_defense_comparison.png")
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")

if __name__ == "__main__":
    # Try to find project root by looking for 'outputs' directory
    if os.path.exists("outputs"):
        base_dir = "."
    elif os.path.exists("../../outputs"):
        base_dir = "../.."
    else:
        base_dir = ".."
    
    plot_dir = os.path.join(base_dir, "outputs/plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("Processing Main Experiments (With Defense)...")
    df_main = load_exp_results("result_consistency_Exp_*.json", base_dir=base_dir)
    
    print("\nProcessing No Defense Experiments...")
    df_no_defense = load_no_defense_results(base_dir=base_dir)
    print(df_no_defense)
    
    if not df_no_defense.empty:
        plot_defense_comparison(df_main, df_no_defense, plot_dir=plot_dir)

    print("\nProcessing Ablation Study...")
    df_ablation = load_ablation_results(base_dir=base_dir)
    print(df_ablation)
    plot_ablation_study(df_ablation, plot_dir=plot_dir)
