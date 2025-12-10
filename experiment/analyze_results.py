import json
import os
import argparse
import glob
from collections import defaultdict
from datetime import datetime

# Optional visualization
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

def analyze_file(file_path):
    print(f"\nAnalyzing: {os.path.basename(file_path)}")
    
    total = 0
    retrieval_hits = 0
    answer_correct = 0
    answer_wrong = 0
    not_evaluated = 0
    
    # Optional: Detailed error analysis storage
    errors = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    total += 1
                    
                    # 1. Retrieval Stats
                    if entry.get("retrieval_success", False):
                        retrieval_hits += 1
                        
                    # 2. Answer Evaluation Stats
                    eval_data = entry.get("evaluation")
                    if eval_data:
                        if eval_data.get("correct", False):
                            answer_correct += 1
                        else:
                            answer_wrong += 1
                            # Store a few errors for sampling
                            if len(errors) < 3: 
                                errors.append({
                                    "id": entry.get("id"),
                                    "query": entry.get("query"),
                                    "prediction": entry.get("prediction"),
                                    "ground_truth": entry.get("ground_truth"),
                                    "reason": eval_data.get("reason")
                                })
                    else:
                        not_evaluated += 1
                        
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {file_path}")
                    continue

        if total == 0:
            print("  No data found.")
            return

        # Calculate Percentages
        retrieval_acc = (retrieval_hits / total) * 100 if total > 0 else 0.0
        # For answer accuracy, we usually base it on total, treating un-evaluated as wrong or separate?
        # Let's base it on evaluated count to be fair, or total. Usually total is safer.
        # But if the run crashed, evaluated might be less than total.
        # Let's use total for Retrieval, and evaluated_total for Answer Acc.
        evaluated_total = answer_correct + answer_wrong
        answer_acc = (answer_correct / evaluated_total * 100) if evaluated_total > 0 else 0.0

        print(f"  Total Entries:      {total}")
        print(f"  Retrieval Success:  {retrieval_hits}/{total} ({retrieval_acc:.2f}%)")
        print(f"  Answer Correct:     {answer_correct}/{evaluated_total} ({answer_acc:.2f}%)")
        if not_evaluated > 0:
            print(f"  Not Evaluated:      {not_evaluated}")

        if errors:
            print("\n  Sample Errors:")
            for e in errors:
                print(f"    [ID {e['id']}] Q: {e['query']}")
                print(f"      GT:   {e['ground_truth']}")
                print(f"      Pred: {e['prediction']}")
                print(f"      Reason: {e['reason']}")
                print("-" * 40)

        return {
            "file": os.path.basename(file_path),
            "total": total,
            "retrieval_hits": retrieval_hits,
            "retrieval_acc": retrieval_acc,
            "evaluated_total": evaluated_total,
            "answer_correct": answer_correct,
            "answer_acc": answer_acc,
            "not_evaluated": not_evaluated,
        }

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def pick_latest_logs():
    logs_dir = "logs"
    if not os.path.isdir(logs_dir):
        return []

    baseline_eval = glob.glob(os.path.join(logs_dir, "baseline_experiment_*_evaluated.jsonl"))
    baseline_eval.sort(key=os.path.getmtime, reverse=True)

    baseline_raw = glob.glob(os.path.join(logs_dir, "baseline_experiment_*.jsonl"))
    baseline_raw = [f for f in baseline_raw if not f.endswith("_evaluated.jsonl")]
    baseline_raw.sort(key=os.path.getmtime, reverse=True)

    edited = glob.glob(os.path.join(logs_dir, "exp_edited_image_*.jsonl"))
    edited = [f for f in edited if os.path.basename(f).startswith("exp_edited_image_") and ("_fake_caption_" not in os.path.basename(f))]
    edited.sort(key=os.path.getmtime, reverse=True)

    edited_fake = glob.glob(os.path.join(logs_dir, "exp_edited_image_fake_caption_*.jsonl"))
    edited_fake.sort(key=os.path.getmtime, reverse=True)

    fake = glob.glob(os.path.join(logs_dir, "exp_fake_caption_*.jsonl"))
    fake.sort(key=os.path.getmtime, reverse=True)

    ordered = []
    if baseline_eval:
        ordered.append(baseline_eval[0])
    if baseline_raw:
        ordered.append(baseline_raw[0])
    if edited:
        ordered.append(edited[0])
    if edited_fake:
        ordered.append(edited_fake[0])
    if fake:
        ordered.append(fake[0])
    return ordered[:4]

def make_visualization(summaries, output_path):
    if not plt:
        print("matplotlib not available; skipping visualization.")
        return

    labels = [s["label"] for s in summaries]
    retrieval = [s["retrieval_acc"] for s in summaries]
    answer = [s["answer_acc"] for s in summaries]

    x = range(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar([i - width/2 for i in x], retrieval, width=width, label="Retrieval HIT %")
    plt.bar([i + width/2 for i in x], answer, width=width, label="Answer CORRECT %")
    plt.xticks(list(x), labels, rotation=15)
    plt.ylim(0, 100)
    plt.ylabel("Percentage (%)")
    plt.title("Poisoning Impact on Multi-modal RAG (Summary)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nSaved visualization: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze experiment result logs.")
    parser.add_argument("file_pattern", nargs="?", default=None, help="File path or glob pattern. If omitted, auto-pick latest four logs.")
    parser.add_argument("--save", dest="save", default=os.path.join("logs", "summary.png"), help="Output path for visualization (PNG)")
    args = parser.parse_args()

    if args.file_pattern:
        files = glob.glob(args.file_pattern)
    else:
        files = pick_latest_logs()

    if not files:
        print("No log files found.")
        return

    print(f"Selected {len(files)} log files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    summaries = []
    for file_path in files:
        res = analyze_file(file_path)
        if not res:
            continue
        label = os.path.basename(file_path)
        summaries.append({
            "label": label,
            **res,
        })

    if summaries:
        make_visualization(summaries, args.save)

if __name__ == "__main__":
    main()
