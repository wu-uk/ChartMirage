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

from core.experiments.pipeline import DualModalPipeline

def main():
    # 1. 配置文件路径
    dataset_path = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    # 使用新的存储目录，避免与旧索引冲突
    storage_dir = os.path.join(base_dir, "storage/storage_dual_experiment")
    
    # 2. 初始化 Pipeline
    pipeline = DualModalPipeline(dataset_path, storage_dir)
    
    # 3. 构建索引 (首次运行会自动创建，后续会加载)
    # 强制重建一次以确保使用了新的“双文档”逻辑
    pipeline.build_index(force_rebuild=True)
    
    # 4. 运行测试查询
    query_str = "what is the trend of sales from 2020 to 2024?"
    response = pipeline.query(query_str)
    
    # 5. 打印回答
    print(f"\nAI 回答:\n{response}\n")
    
    # 6. 验证检索内容
    pipeline.verify_retrieval(response)

if __name__ == "__main__":
    main()
