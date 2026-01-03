import os

# Try to find project root by looking for 'dataset' directory
if os.path.exists("dataset"):
    base_dir = "."
elif os.path.exists("../../dataset"):
    base_dir = "../.."
else:
    base_dir = ".."

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.core.schema import ImageNode, TextNode

# 1. 设置 API Key
load_dotenv()
if not os.environ["OPENAI_API_KEY"].startswith("sk-"):
    print("请设置环境变量!!!")

print("正在初始化模型...")

# 2. 初始化多模态模型
# 使用 OpenAI Embedding 处理文本
text_embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# 使用 CLIP 将图片变成向量，用于检索
image_embed_model = ClipEmbedding(model_name="ViT-L/14")

# 使用 Qwen3-VL-Plus 理解图片和回答问题
openai_mm_llm = OpenAILike(
    model="qwen3-vl-plus", 
    is_chat_model=True,
    is_function_calling_model=True
)

# 3. 加载数据
print("正在加载图表数据...")
data_dir = os.path.join(base_dir, "data_charts_gen")
documents = SimpleDirectoryReader(data_dir).load_data()

# 4. 构建多模态索引 (Index)
print("正在构建索引...")
index = MultiModalVectorStoreIndex.from_documents(
    documents,
    embed_model=text_embed_model,
    image_embed_model=image_embed_model,
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
print("【验证检索内容：图片 vs 文字】")

if not response.source_nodes:
    print("❌ 未检索到任何内容。")
else:
    for i, node_with_score in enumerate(response.source_nodes):
        node = node_with_score.node
        print(f"\n[Node {i+1}]")
        print(f"  节点类: {type(node).__name__}")
        print(f"  相似度分数 (Score): {node_with_score.score:.4f}")
        
        if isinstance(node, ImageNode):
            print("  -> 🖼️  这是一个图片节点 (ImageNode)")
            print(f"  -> 图片路径: {node.metadata.get('file_path') or node.image_path}")
        elif isinstance(node, TextNode):
            print("  -> 📝 这是一个文本节点 (TextNode)")
            print(f"  -> 文本内容摘要: {node.text[:50]}...")
        else:
            print(f"  -> 其他类型: {node}")

    has_image = any(isinstance(n.node, ImageNode) for n in response.source_nodes)
    # Strict check for TextNode (excluding ImageNode which inherits from TextNode)
    has_text = any(isinstance(n.node, TextNode) and not isinstance(n.node, ImageNode) for n in response.source_nodes)

    print("-" * 30)
    if has_image and has_text:
        print("✅ 结论: 系统同时检索到了图片和纯文本节点。")
    elif has_image:
        print("⚠️ 结论: 仅检索到了图片节点。")
    elif has_text:
        print("⚠️ 结论: 仅检索到了纯文本节点。")