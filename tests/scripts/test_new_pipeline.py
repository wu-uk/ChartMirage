import os
import sys

# Try to find project root by looking for 'dataset' directory
if os.path.exists("dataset"):
    base_dir = "."
elif os.path.exists("../../dataset"):
    base_dir = "../.."
else:
    base_dir = ".."

# Add project root to sys.path
sys.path.append(os.path.abspath(base_dir))
# Add 'core' to sys.path
sys.path.append(os.path.abspath(os.path.join(base_dir, "core")))

from core.experiments.pipeline import DualModalPipeline, ImagePipeline
from defensive_rag_pipeline import DefensiveRAGPipeline

def main():
    # 1. 配置
    dataset_path = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    
    # === 测试 DualModalPipeline ===
    print("\n" + "="*50)
    print("Testing DualModalPipeline (Image + Text Docs)")
    print("="*50)
    
    dual_storage = os.path.join(base_dir, "storage/storage_dual_experiment")
    # 开启评估功能
    dual_pipeline = DualModalPipeline(dataset_path, dual_storage, eval_enabled=True)
    dual_pipeline.build_index(force_rebuild=False) # 如果之前构建过可以设为False
    
    query_str = "what is the trend of sales from 2020 to 2024?"
    # 这里我们模拟一个 Ground Truth，实际使用时从 dataset 获取
    ground_truth = "The sales decreased from 2020 to 2024." 
    
    response, eval_res = dual_pipeline.query(query_str, ground_truth)
    print(f"\nAI Answer: {response}")

    # === 测试 ImagePipeline ===
    print("\n" + "="*50)
    print("Testing ImagePipeline (Only ImageDocument)")
    print("="*50)
    
    image_storage = os.path.join(base_dir, "storage/storage_image_experiment")
    img_pipeline = ImagePipeline(dataset_path, image_storage, eval_enabled=True)
    img_pipeline.build_index(force_rebuild=False)
    
    response_img, eval_res_img = img_pipeline.query(query_str, ground_truth)
    print(f"\nAI Answer: {response_img}")

if __name__ == "__main__":
    main()
