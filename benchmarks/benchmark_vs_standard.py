
print("Script started...")
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

import json

# Set environment variables for China network and PaddleOCR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

import json
import random
import torch
import numpy as np
import pandas as pd
import clip
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

print("Imports done (part 1)...")

from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv

print("Importing Ragas...")
from ragas.metrics import faithfulness, Faithfulness
from ragas import evaluate
from datasets import Dataset
from langchain_openai import ChatOpenAI

print("Importing Pipeline...")
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

print("Initialization complete.")

# Load environment variables
load_dotenv()

class BaselineA:
    """Embedding Guardrail Baseline using CLIP similarity (Local)"""
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.lock = threading.Lock()
        print("Loading Baseline A (CLIP Local)...")
        # Reuse the same model as in DefensiveRAGPipeline to avoid downloads
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.threshold = 0.25 

    def check(self, image_path, text_content):
        try:
            with self.lock:
                image = Image.open(image_path).convert("RGB")
                image_input = self.preprocess(image).unsqueeze(0).to(self.device)
                text_input = clip.tokenize([text_content], truncate=True).to(self.device)
                
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    text_features = self.model.encode_text(text_input)
                    
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    
                    similarity = (image_features @ text_features.T).item()
            
            is_safe = similarity > self.threshold
            return is_safe, similarity
        except Exception as e:
            return True, 0.0

class BaselineB:
    """RAGAS Faithfulness Baseline"""
    def __init__(self, vlm_pipeline):
        print("Initializing Baseline B (RAGAS)...")
        self.vlm = vlm_pipeline.vlm # Reuse the VLM from your pipeline
        self.llm = ChatOpenAI(
            model="qwen3-vl-plus",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            timeout=60,
            max_retries=2,
        )
        # Manually initialize the metric instance with the LLM
        self.metric = Faithfulness(llm=self.llm)
        self.threshold = 0.7 # Faithfulness threshold
        self.description_cache = {}
        self.cache_lock = threading.Lock()

    def _get_vlm_description(self, image_path):
        # 检查缓存以避免重复生成相同图片的描述
        with self.cache_lock:
            if image_path in self.description_cache:
                return self.description_cache[image_path]

        # Use VLM to get a detailed description of the image as context
        from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock
        prompt = "Describe this chart in detail, including all labels, trends, and connections."
        message = ChatMessage(role="user", blocks=[TextBlock(text=prompt), ImageBlock(path=image_path)])
        try:
            # The VLM call itself is already thread-safe in DefensiveRAGPipeline via gpu_lock if needed, 
            # but OpenAILike is just an API wrapper, so it's fine.
            response = self.vlm.chat([message])
            desc = str(response).strip()
            
            with self.cache_lock:
                self.description_cache[image_path] = desc
            return desc
        except Exception as e:
            print(f"VLM Description error: {e}")
            return "Error generating description"

    def check(self, image_path, text_content):
        from ragas.dataset_schema import SingleTurnSample
        try:
            description = self._get_vlm_description(image_path)
            sample = SingleTurnSample(
                user_input="Analyze the chart consistency",
                retrieved_contexts=[description],
                response=text_content
            )
            # 移除全局锁，允许并发 API 调用
            score = self.metric.single_turn_score(sample)
            
            if score is None or (isinstance(score, float) and np.isnan(score)):
                score = 0.0
                
            # Faithfulness score ranges from 0 to 1. 
            # Low score means unfaithful (potential attack).
            is_safe = score >= self.threshold
            return is_safe, score
        except Exception as e:
            # print(f"Baseline B check error: {e}")
            return True, 0.0

def prepare_data(dataset_path, num_samples=100):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    samples = data[:num_samples]
    
    test_set = []
    
    # 1. Normal Samples (Clean Image + Clean Text)
    for i in range(len(samples) // 2):
        item = samples[i]
        test_set.append({
            "image_path": os.path.join("dataset", item["image"]),
            "text": item["original_caption"],
            "label": "SAFE",
            "type": "Normal"
        })
    
    # 2. Attack Samples (Clean Image + Fake Text)
    for i in range(len(samples) // 2, (len(samples) // 2) + (num_samples // 4)):
        item = samples[i]
        test_set.append({
            "image_path": os.path.join("dataset", item["image"]),
            "text": item["fake_caption"],
            "label": "UNSAFE",
            "type": "Semantic Attack (Fake Text)"
        })
        
    # 3. Attack Samples (Fake Image + Clean Text)
    for i in range((len(samples) // 2) + (num_samples // 4), num_samples):
        item = samples[i]
        test_set.append({
            "image_path": os.path.join("dataset", item["edited_image"]),
            "text": item["original_caption"],
            "label": "UNSAFE",
            "type": "Semantic Attack (Fake Image)"
        })
        
    return test_set

def evaluate_sample(item, my_pipeline, baseline_a, baseline_b):
    img_path = item["image_path"]
    text = item["text"]
    true_label = item["label"]
    
    try:
        # My Pipeline
        my_safe, my_level, my_msg = my_pipeline.process(img_path, text)
        my_pred = "SAFE" if my_safe else "UNSAFE"
        
        # Baseline A
        a_safe, a_score = baseline_a.check(img_path, text)
        a_pred = "SAFE" if a_safe else "UNSAFE"
        
        # Baseline B
        b_safe, b_score = baseline_b.check(img_path, text)
        b_pred = "SAFE" if b_safe else "UNSAFE"
        
        return {
            "Type": item["type"],
            "True Label": true_label,
            "My Pred": my_pred,
            "A Pred": a_pred,
            "B Pred": b_pred
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def run_benchmark():
    dataset_path = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    print(f"Preparing benchmark data from {dataset_path}...")
    test_set = prepare_data(dataset_path, num_samples=200) 
    print(f"Total samples to test: {len(test_set)}")
    
    partial_results_path = os.path.join(base_dir, "outputs/results/benchmark_partial.jsonl")
    os.makedirs(os.path.dirname(partial_results_path), exist_ok=True)
    finished_samples = {}
    if os.path.exists(partial_results_path):
        with open(partial_results_path, 'r') as f:
            for line in f:
                try:
                    res = json.loads(line)
                    key = (res["image_path"], res["text"])
                    finished_samples[key] = res
                except: continue
    print(f"Found {len(finished_samples)} already finished samples.")

    # Initialize Pipelines
    print("Initializing DefensiveRAGPipeline (L1-L3)...")
    my_pipeline = DefensiveRAGPipeline()
    print("Initializing Baseline A (CLIP)...")
    baseline_a = BaselineA()
    print("Initializing Baseline B (Ragas)...")
    baseline_b = BaselineB(my_pipeline)
    
    results = list(finished_samples.values())
    remaining_samples = [item for item in test_set if (item["image_path"], item["text"]) not in finished_samples]
    
    if remaining_samples:
        print(f"\nStarting Parallel Benchmark Loop ({len(remaining_samples)} remaining, max_workers=20)...")
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(evaluate_sample, item, my_pipeline, baseline_a, baseline_b): item for item in remaining_samples}
            
            with open(partial_results_path, 'a') as f_out:
                iterator = as_completed(futures)
                for _ in tqdm(range(len(remaining_samples)), desc="Benchmarking"):
                    try:
                        future = next(iterator)
                        item = futures[future]
                        res = future.result(timeout=180) # 3 min timeout per sample
                        if res:
                            res["image_path"] = item["image_path"]
                            res["text"] = item["text"]
                            results.append(res)
                            f_out.write(json.dumps(res) + "\n")
                            f_out.flush()
                    except StopIteration: break
                    except Exception as e:
                        print(f"\nSample failed or timed out: {e}")
                        continue

    if not results:
        print("No results to analyze.")
        return

    # Analysis
    df = pd.DataFrame(results)
    
    def calculate_metrics(df, pred_col):
        # SAFE is Negative, UNSAFE is Positive (Attack detection)
        # FP: True SAFE, Pred UNSAFE
        # FN: True UNSAFE, Pred SAFE
        
        safe_mask = df["True Label"] == "SAFE"
        unsafe_mask = df["True Label"] == "UNSAFE"
        
        fp = ((df[pred_col] == "UNSAFE") & safe_mask).sum()
        fn = ((df[pred_col] == "SAFE") & unsafe_mask).sum()
        
        total_safe = safe_mask.sum()
        total_unsafe = unsafe_mask.sum()
        
        fpr = fp / total_safe if total_safe > 0 else 0
        fnr = fn / total_unsafe if total_unsafe > 0 else 0
        
        return fpr, fnr

    my_fpr, my_fnr = calculate_metrics(df, "My Pred")
    a_fpr, a_fnr = calculate_metrics(df, "A Pred")
    b_fpr, b_fnr = calculate_metrics(df, "B Pred")
    
    # Scenario breakdown: Failure percentage in UNSAFE scenarios
    unsafe_df = df[df["True Label"] == "UNSAFE"]
    my_fail = (unsafe_df["My Pred"] == "SAFE").mean() * 100
    a_fail = (unsafe_df["A Pred"] == "SAFE").mean() * 100
    b_fail = (unsafe_df["B Pred"] == "SAFE").mean() * 100

    print("\n" + "="*50)
    print("BENCHMARK RESULTS: Hierarchical vs Standard")
    print("="*50)
    
    report_data = {
        "Metric": ["False Positive Rate (FPR)", "False Negative Rate (FNR)", "Failure Rate (Semantic Conflict)"],
        "My Pipeline (L1-L3)": [f"{my_fpr:.2%}", f"{my_fnr:.2%}", f"{my_fail:.1f}%"],
        "Baseline A (CLIP-Sim)": [f"{a_fpr:.2%}", f"{a_fnr:.2%}", f"{a_fail:.1f}%"],
        "Baseline B (Ragas Faith)": [f"{b_fpr:.2%}", f"{b_fnr:.2%}", f"{b_fail:.1f}%"]
    }
    
    report_df = pd.DataFrame(report_data)
    print(report_df.to_string(index=False))
    
    print("\nDetailed Scenario Breakdown (Failure %):")
    scenarios = df["Type"].unique()
    breakdown = []
    for sc in scenarios:
        if sc == "Normal": continue
        sc_df = df[df["Type"] == sc]
        breakdown.append({
            "Scenario": sc,
            "My Fail": (sc_df["My Pred"] == "SAFE").mean() * 100,
            "A Fail": (sc_df["A Pred"] == "SAFE").mean() * 100,
            "B Fail": (sc_df["B Pred"] == "SAFE").mean() * 100
        })
    
    breakdown_df = pd.DataFrame(breakdown)
    print(breakdown_df.to_string(index=False))
    
    # Save raw results for plotting
    raw_results_path = os.path.join(base_dir, "outputs/results/benchmark_raw_results.csv")
    df.to_csv(raw_results_path, index=False)
    print(f"\nRaw results saved to {raw_results_path}")
    
    # Save summary results
    summary_path = os.path.join(base_dir, "outputs/results/benchmark_summary.csv")
    report_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    run_benchmark()
