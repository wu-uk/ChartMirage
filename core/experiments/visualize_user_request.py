import json
import os
import matplotlib.pyplot as plt
import numpy as np
import glob

def calculate_stats(data):
    total = len(data)
    if total == 0:
        return 0, 0, 0

    refusals = 0
    correct = 0
    attack_success = 0
    
    for item in data:
        # Check for refusal
        prediction = item.get("prediction", "")
        if "Refused to answer" in prediction or item.get("is_refusal", False):
            refusals += 1
        else:
            # Check accuracy
            eval_res = item.get("eval_result", {})
            if eval_res.get("correct"):
                correct += 1
            
            # Check ASR
            # Prefer 'asr_evaluation' object, then 'asr_result', then manual check if fake_answer exists
            asr_eval = item.get("asr_evaluation") or item.get("asr_result")
            if asr_eval and isinstance(asr_eval, dict):
                if asr_eval.get("match"):
                    attack_success += 1
            elif item.get("fake_answer"):
                # Fallback if no pre-calculated ASR but fake_answer exists (though we prefer pre-calc)
                pass

    refusal_rate = (refusals / total) * 100
    accuracy = (correct / total) * 100
    asr = (attack_success / total) * 100
    
    return refusal_rate, accuracy, asr

def load_json_or_jsonl(file_path):
    data = []
    try:
        if file_path.endswith(".jsonl"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data

def main():
    # Try to find project root by looking for 'dataset' directory
    if os.path.exists("dataset"):
        base_dir = "."
    elif os.path.exists("../../dataset"):
        base_dir = "../.."
    else:
        base_dir = ".."

    # 1. Main Experiments (No Defense) - User provided paths
    main_files = [
        os.path.join(base_dir, "outputs/results/results_Exp_1_Baseline_Orig_Image_+_Orig_Caption_asr_evaluated.jsonl"),
        os.path.join(base_dir, "outputs/results/results_Exp_2_Caption_Conflict_Orig_Image_+_Fake_Caption_asr_evaluated.jsonl"),
        os.path.join(base_dir, "outputs/results/results_Exp_3_Image_Conflict_Fake_Image_+_Orig_Caption_asr_evaluated.jsonl"),
        os.path.join(base_dir, "outputs/results/results_Exp_4_Aligned_Fake_Fake_Image_+_Fake_Caption_asr_evaluated.jsonl")
    ]
    
    # 2. Noise Experiments - Use ASR evaluated files
    noise_files = [
        os.path.join(base_dir, "outputs/results/results_Noise_gaussian_blur_asr_evaluated.jsonl"),
        os.path.join(base_dir, "outputs/results/results_Noise_gaussian_noise_asr_evaluated.jsonl"),
        os.path.join(base_dir, "outputs/results/results_Noise_rotation_asr_evaluated.jsonl"),
        os.path.join(base_dir, "outputs/results/results_Noise_salt_pepper_noise_asr_evaluated.jsonl")
    ]
    
    # 3. Output directory for plots
    plot_dir = os.path.join(base_dir, "outputs/plots")
    os.makedirs(plot_dir, exist_ok=True)

    
    # Data containers
    main_results = []
    noise_results = []
    
    # Process Main Experiments
    print("Processing Main Experiments (No Defense)...")
    for fp in main_files:
        if not os.path.exists(fp):
            print(f"Warning: File not found {fp}")
            continue
            
        data = load_json_or_jsonl(fp)
        ref, acc, asr = calculate_stats(data)
        
        # Extract readable name
        name = os.path.basename(fp)
        if "Exp_1" in name: label = "Baseline"
        elif "Exp_2" in name: label = "Text Poison"
        elif "Exp_3" in name: label = "Image Poison"
        elif "Exp_4" in name: label = "Dual Poison"
        else: label = name
        
        main_results.append({
            "Label": label,
            "Accuracy": acc,
            "ASR": asr,
            "Refusal": ref
        })
        print(f"  {label}: Acc={acc:.2f}%, ASR={asr:.2f}%")

    # Process Noise Experiments
    print("\nProcessing Noise Experiments...")
    for fp in noise_files:
        if not os.path.exists(fp):
            print(f"Warning: File not found {fp}")
            continue
            
        data = load_json_or_jsonl(fp)
        ref, acc, asr = calculate_stats(data)
        
        # Extract readable name
        name = os.path.basename(fp).replace("results_Noise_", "").replace("_asr_evaluated.jsonl", "")
        label = name.replace("_", " ").title()
        
        noise_results.append({
            "Label": label,
            "Accuracy": acc,
            "ASR": asr,
            "Refusal": ref
        })
        print(f"  {label}: Acc={acc:.2f}%, ASR={asr:.2f}%")

    # Plotting
    plot_charts(main_results, noise_results, plot_dir)

def plot_charts(main_data, noise_data, plot_dir):
    # Chart 1: Main Experiments (No Defense)
    if main_data:
        labels = [d["Label"] for d in main_data]
        accs = [d["Accuracy"] for d in main_data]
        asrs = [d["ASR"] for d in main_data]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='#55a868')
        rects2 = ax.bar(x + width/2, asrs, width, label='ASR', color='#c44e52')
        
        ax.set_ylabel('Percentage (%)')
        ax.set_title('No Defense Performance (Main Experiments)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        ax.bar_label(rects1, fmt='%.1f', padding=3)
        ax.bar_label(rects2, fmt='%.1f', padding=3)
        
        plt.tight_layout()
        save_path = os.path.join(plot_dir, "plot_no_defense_main.png")
        plt.savefig(save_path, dpi=300)
        print(f"\nSaved {save_path}")
    
    # Chart 2: Noise Experiments
    if noise_data:
        labels = [d["Label"] for d in noise_data]
        accs = [d["Accuracy"] for d in noise_data]
        asrs = [d["ASR"] for d in noise_data]
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='#55a868')
        rects2 = ax.bar(x + width/2, asrs, width, label='ASR', color='#c44e52')
            
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Robustness against Noise (Accuracy & ASR)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.legend()
        
        ax.bar_label(rects1, fmt='%.1f', padding=3)
        ax.bar_label(rects2, fmt='%.1f', padding=3)
        
        plt.tight_layout()
        save_path = os.path.join(plot_dir, "plot_noise_robustness.png")
        plt.savefig(save_path, dpi=300)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
