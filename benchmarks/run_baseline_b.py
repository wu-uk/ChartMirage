
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
import random
import torch
import numpy as np
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from ragas.metrics.collections import Faithfulness
from datasets import Dataset
from langchain_openai import ChatOpenAI
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

# Set environment variables for China network
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

load_dotenv()

class BaselineB:
    """RAGAS Faithfulness Baseline"""
    def __init__(self, vlm_pipeline):
        print("Initializing Baseline B (RAGAS)...")
        self.vlm = vlm_pipeline.vlm 
        self.llm = ChatOpenAI(
            model="qwen3-vl-plus",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            timeout=60, # 增加 60 秒超时
            max_retries=2,
        )
        self.metric = Faithfulness(llm=self.llm)
        self.threshold = 0.7 
        self.description_cache = {}
        self.cache_lock = threading.Lock()

    def _get_vlm_description(self, image_path):
        # 检查缓存以避免重复生成相同图片的描述
        with self.cache_lock:
            if image_path in self.description_cache:
                return self.description_cache[image_path]

        from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock
        prompt = "Describe this chart in detail, including all labels, trends, and connections."
        message = ChatMessage(role="user", blocks=[TextBlock(text=prompt), ImageBlock(path=image_path)])
        try:
            response = self.vlm.chat([message])
            desc = str(response).strip()
            
            with self.cache_lock:
                self.description_cache[image_path] = desc
            return desc
        except Exception as e:
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
                
            # 忠实度分越低越危险，阈值 0.5。score < 0.5 判定为 UNSAFE (is_safe=False)
            is_safe = score >= self.threshold
            return is_safe, score
        except Exception as e:
            # 发生错误时保守起见，暂时返回 SAFE 以便后续查看报错，但记录分数为 0
            return True, 0.0

def prepare_data(dataset_path, num_samples=200):
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    random.seed(42)
    random.shuffle(data)
    samples = data[:num_samples]
    test_set = []
    
    # 1. Normal Samples
    for i in range(len(samples) // 2):
        item = samples[i]
        test_set.append({"image_path": os.path.join(base_dir, "dataset", item["image"]), "text": item["original_caption"], "label": "SAFE", "type": "Normal"})
    
    # 2. Attack Samples (Fake Text)
    for i in range(len(samples) // 2, (len(samples) // 2) + (num_samples // 4)):
        item = samples[i]
        test_set.append({"image_path": os.path.join(base_dir, "dataset", item["image"]), "text": item["fake_caption"], "label": "UNSAFE", "type": "Semantic Attack (Fake Text)"})
        
    # 3. Attack Samples (Fake Image)
    for i in range((len(samples) // 2) + (num_samples // 4), num_samples):
        item = samples[i]
        test_set.append({"image_path": os.path.join(base_dir, "dataset", item["edited_image"]), "text": item["original_caption"], "label": "UNSAFE", "type": "Semantic Attack (Fake Image)"})
    return test_set

def evaluate_baseline_b(item, baseline_b):
    img_path = item["image_path"]
    text = item["text"]
    true_label = item["label"]
    try:
        b_safe, b_score = baseline_b.check(img_path, text)
        b_pred = "SAFE" if b_safe else "UNSAFE"
        return {"Type": item["type"], "True Label": true_label, "B Pred": b_pred, "B Score": b_score}
    except Exception as e:
        return None

def run_standalone_b():
    dataset_path = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    test_set = prepare_data(dataset_path, num_samples=200)
    print(f"Total samples to test for Baseline B: {len(test_set)}")
    
    partial_results_path = os.path.join(base_dir, "outputs/results/baseline_b_partial.jsonl")
    os.makedirs(os.path.dirname(partial_results_path), exist_ok=True)
    finished_samples = {}
    if os.path.exists(partial_results_path):
        with open(partial_results_path, 'r') as f:
            for line in f:
                try:
                    res = json.loads(line)
                    # Use (image_path, text) as key
                    key = (res["image_path"], res["text"])
                    finished_samples[key] = res
                except:
                    continue
    print(f"Found {len(finished_samples)} already finished samples.")

    my_pipeline = DefensiveRAGPipeline()
    baseline_b = BaselineB(my_pipeline)
    
    results = list(finished_samples.values())
    remaining_samples = [item for item in test_set if (item["image_path"], item["text"]) not in finished_samples]
    
    if not remaining_samples:
        print("All samples already processed.")
    else:
        print(f"\nStarting Parallel Benchmark for Baseline B ({len(remaining_samples)} remaining, max_workers=20)...")
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {executor.submit(evaluate_baseline_b, item, baseline_b): item for item in remaining_samples}
            
            with open(partial_results_path, 'a') as f_out:
                # 使用 timeout 防止个别任务导致整体挂起
                iterator = as_completed(futures)
                for _ in tqdm(range(len(remaining_samples)), desc="Processing"):
                    try:
                        # 每一个任务最多等待 120 秒
                        future = next(iterator)
                        item = futures[future]
                        res = future.result(timeout=120) 
                        if res:
                            res["image_path"] = item["image_path"]
                            res["text"] = item["text"]
                            results.append(res)
                            f_out.write(json.dumps(res) + "\n")
                            f_out.flush()
                    except StopIteration:
                        break
                    except Exception as e:
                        print(f"\nTask failed or timed out: {e}")
                        continue

    if not results:
        print("No results to analyze.")
        return

    df = pd.DataFrame(results)
    safe_mask = df["True Label"] == "SAFE"
    unsafe_mask = df["True Label"] == "UNSAFE"
    fp = ((df["B Pred"] == "UNSAFE") & safe_mask).sum()
    fn = ((df["B Pred"] == "SAFE") & unsafe_mask).sum()
    fpr = fp / safe_mask.sum() if safe_mask.sum() > 0 else 0
    fnr = fn / unsafe_mask.sum() if unsafe_mask.sum() > 0 else 0
    
    print("\n" + "="*50)
    print("STANDALONE BASELINE B RESULTS")
    print("="*50)
    print(f"False Positive Rate (FPR): {fpr:.2%}")
    print(f"False Negative Rate (FNR): {fnr:.2%}")
    
    df.to_csv("baseline_b_raw_results.csv", index=False)
    print("\nRaw results saved to baseline_b_raw_results.csv")

if __name__ == "__main__":
    run_standalone_b()
