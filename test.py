import os
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.llms.openai import OpenAI
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.clip import ClipEmbedding

# 1. 设置 API Key (或者在环境变量中设置)
os.environ["OPENAI_API_KEY"] = "sk-7RQMk5JJvBU94F9RzdLB7Le3YXVlQbxBZjMEmLUPBBrxfu7u" 

print("正在初始化模型...")

# 2. 初始化多模态模型
# 使用 CLIP 将图片变成向量，用于检索
embed_model = ClipEmbedding(model_name="ViT-L/14")

# 使用 GPT-4o 理解图片和回答问题
openai_mm_llm = OpenAI(
    model_name="gpt-4o-mini", 
    api_base="https://www.dmxapi.cn/v1",
    api_key=os.environ["OPENAI_API_KEY"], 
)

# 3. 加载数据
print("正在加载图表数据...")
documents = SimpleDirectoryReader("./data_charts_gen").load_data()

# 4. 构建多模态索引 (Index)
print("正在构建索引...")
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    image_vector_store=None # 默认使用内存存储，适合实验
)

# 5. 构建检索引擎 (Retriever Engine)
retriever_engine = index.as_query_engine(
    llm=openai_mm_llm,
    similarity_top_k=1,
    image_similarity_top_k=1
)

# --- 开始实验 ---

query_str = "what is the trend of sales from 2020 to 2024?"

print(f"\n用户提问: {query_str}")
print("-" * 30)

# 执行 RAG 流程
response = retriever_engine.query(query_str)

# --- 结果分析 ---

print(f"AI 回答:\n{response}\n")

print("-" * 30)
print("【研究核心指标：检索情况】")

for node in response.source_nodes:
    print(f"检索到的图片路径: {node.metadata.get('file_path')}")
    print(f"检索相似度分数 (Score): {node.score:.4f}") 
    # 如果投毒成功，这个 Score 应该显著下降，或者检索到错误的图片