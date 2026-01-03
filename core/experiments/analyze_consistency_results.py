import json
import os
import glob

def analyze_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    except json.JSONDecodeError:
        print(f"Skipping {os.path.basename(file_path)}: JSON decode error (likely incomplete).")
        return None

    total = len(results)
    if total == 0:
        return None

    refusals = 0
    correct = 0
    attack_success = 0
    errors = 0

    for res in results:
        if res.get("error"):
            errors += 1
            continue
            
        if res.get("is_refusal"):
            refusals += 1
        else:
            # Accuracy
            if res.get("eval_result", {}).get("correct"):
                correct += 1
            
            # ASR
            if res.get("asr_result", {}).get("match"):
                attack_success += 1

    refusal_rate = (refusals / total) * 100
    accuracy = (correct / total) * 100
    asr = (attack_success / total) * 100
    
    return {
        "Experiment": os.path.basename(file_path).replace("result_consistency_", "").replace(".json", ""),
        "Total": total,
        "Refusals": f"{refusals} ({refusal_rate:.2f}%)",
        "Accuracy": f"{correct} ({accuracy:.2f}%)",
        "ASR": f"{attack_success} ({asr:.2f}%)"
    }

def main():
    files = sorted(glob.glob("result_consistency_Exp_*.json"))
    summary = []
    
    for f in files:
        # Avoid the old small exp4 file
        if "exp4" in f.lower() and "Dual_Poison" not in f:
             continue
             
        res = analyze_file(f)
        if res:
            summary.append(res)
    
    print("\n" + "="*90)
    print(f"{'Experiment':<40} | {'Refusals':<18} | {'Accuracy':<18} | {'ASR':<10}")
    print("-" * 90)
    for s in summary:
        print(f"{s['Experiment']:<40} | {s['Refusals']:<18} | {s['Accuracy']:<18} | {s['ASR']:<10}")
    print("="*90)

if __name__ == "__main__":
    main()
