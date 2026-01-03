import os
import sys

# 注入 CUDA 库路径以解决 PaddleOCR 的 libnvrtc.so.13 缺失问题
cuda_lib_path = "/home/ASC26team2/miniconda3/envs/ChartMirage/lib/python3.12/site-packages/nvidia/cu13/lib"
if os.path.exists(cuda_lib_path):
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if cuda_lib_path not in current_ld_path:
        os.environ["LD_LIBRARY_PATH"] = cuda_lib_path + ":" + current_ld_path
        # 重新执行当前脚本，使环境变量生效
        try:
            os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
        except Exception as e:
            print(f"Failed to re-execute script: {e}")

os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

import json
import time
import torch
import threading
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
# Try to find project root by looking for 'dataset' directory
if os.path.exists("dataset"):
    base_dir = "."
elif os.path.exists("../../dataset"):
    base_dir = "../.."
else:
    base_dir = ".."

# Add 'core' to sys.path so we can import defensive_rag_pipeline
sys.path.append(os.path.abspath(os.path.join(base_dir, "core")))

from defensive_rag_pipeline import DefensiveRAGPipeline

class AblationPipeline(DefensiveRAGPipeline):
    def __init__(self, enabled_layers=(1, 2, 3), **kwargs):
        super().__init__(**kwargs)
        self.enabled_layers = enabled_layers
        self.l3_calls = 0
        self.total_tokens = 0
        self.token_lock = threading.Lock()

    def run_level3(self, image_path, text_content, query="Is the image content consistent with the text description?"):
        with self.token_lock:
            self.l3_calls += 1
        
        # Call the original run_level3 but we'll try to estimate tokens 
        # since the wrapper might not expose it easily in all environments
        # Typical Qwen-VL usage: ~1000 tokens per image + prompt tokens
        res = super().run_level3(image_path, text_content, query)
        
        # Heuristic estimation if usage is not returned: 
        # Image (1024) + Prompt (~200) + Completion (~100) = ~1324 tokens
        with self.token_lock:
            self.total_tokens += 1324 
        return res

    def process(self, image_path, text_content, query=None):
        # Reset is handled per run_config, but let's ensure layers are respected
        # Level 1
        if 1 in self.enabled_layers:
            l1_safe, l1_msg = self.run_level1(image_path, text_content)
            if l1_safe:
                return True, "Level 1", "Information verified (Trusted source)"
        
        # Level 2
        if 2 in self.enabled_layers:
            l2_safe, l2_msg = self.run_level2(image_path, text_content)
            if not l2_safe:
                return False, "Level 2", f"Refused: {l2_msg}"
        else:
            # If L2 is disabled but L3 is enabled, we need a "safe" pass to reach L3
            # In our pipeline, L2 failure stops the process. If L2 is skipped, we proceed.
            pass

        # Level 3
        if 3 in self.enabled_layers:
            l3_safe, l3_reason, l3_response = self.run_level3(image_path, text_content, query)
            if not l3_safe:
                return False, "Level 3", f"Refused: {l3_reason}"
            return True, "Level 3", l3_response
        
        # If L3 is disabled and we reached here (either L2 passed or was skipped)
        return True, "None/L2", "Proceeded without L3"

def prepare_data(dataset_path, num_samples=200):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    import random
    random.seed(42)
    random.shuffle(data)
    samples = data[:num_samples]
    test_set = []
    # 100 Safe, 50 Fake Text, 50 Fake Image
    for i in range(100):
        item = samples[i]
        test_set.append({"image_path": os.path.join(base_dir, "dataset", item["image"]), "text": item["original_caption"], "label": "SAFE", "type": "Normal"})
    for i in range(100, 150):
        item = samples[i]
        test_set.append({"image_path": os.path.join(base_dir, "dataset", item["image"]), "text": item["fake_caption"], "label": "UNSAFE", "type": "Fake Text"})
    for i in range(150, 200):
        item = samples[i]
        test_set.append({"image_path": os.path.join(base_dir, "dataset", item["edited_image"]), "text": item["original_caption"], "label": "UNSAFE", "type": "Fake Image"})
    return test_set


def run_config(config_name, layers, test_set):
    print(f"\n>>> Testing Configuration: {config_name} (Layers: {layers})")
    pipeline = AblationPipeline(enabled_layers=layers)
    
    results = []
    start_time = time.time()
    
    def eval_item(item):
        s_t = time.time()
        safe, level, msg = pipeline.process(item["image_path"], item["text"])
        e_t = time.time()
        pred = "SAFE" if safe else "UNSAFE"
        return {
            "pred": pred,
            "true": item["label"],
            "latency": e_t - s_t
        }

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(eval_item, item) for item in test_set]
        for f in tqdm(as_completed(futures), total=len(test_set), desc=config_name):
            results.append(f.result())
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    df = pd.DataFrame(results)
    fpr = len(df[(df['true'] == 'SAFE') & (df['pred'] == 'UNSAFE')]) / len(df[df['true'] == 'SAFE'])
    fnr = len(df[(df['true'] == 'UNSAFE') & (df['pred'] == 'SAFE')]) / len(df[df['true'] == 'UNSAFE'])
    avg_latency = df['latency'].mean()
    
    return {
        "Config": config_name,
        "Layers": str(layers),
        "FPR (%)": fpr * 100,
        "FNR (%)": fnr * 100,
        "Avg Latency (s)": avg_latency,
        "VLM Calls": pipeline.l3_calls,
        "Est. Tokens": pipeline.total_tokens,
        "Total Time (s)": total_time
    }

if __name__ == "__main__":
    test_set = prepare_data(os.path.join(base_dir, "dataset/final_qa_merged_unified.json"))
    
    configs = [
        ("Full (L1+L2+L3)", (1, 2, 3)),
        ("No L1 (L2+L3)", (2, 3)),
        ("No L2 (L1+L3)", (1, 3)),
        ("No L3 (L1+L2)", (1, 2)),
        ("L1 Only", (1,)),
        ("L2 Only", (2,)),
        ("L3 Only", (3,))
    ]
    
    all_results = []
    for name, layers in configs:
        res = run_config(name, layers, test_set)
        all_results.append(res)
    
    summary_df = pd.DataFrame(all_results)
    results_path = os.path.join(base_dir, "outputs/results/ablation_results.csv")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    summary_df.to_csv(results_path, index=False)
    print(f"Ablation study completed. Results saved to {results_path}")
    print("\n" + "="*50)
    print("ABLATION STUDY SUMMARY")
    print("="*50)
    print(summary_df.to_string(index=False))
