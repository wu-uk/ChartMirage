import os
import json
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import ImageDocument, TextNode, ImageNode
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class BasePipeline:
    def __init__(self, dataset_path, storage_dir, eval_enabled=True):
        """
        基础 Pipeline 类
        :param dataset_path: 数据集 JSON 路径
        :param storage_dir: 索引持久化存储路径
        :param eval_enabled: 是否开启评估
        """
        load_dotenv()
        self.dataset_path = dataset_path
        self.storage_dir = storage_dir
        self.eval_enabled = eval_enabled
        
        # 1. 初始化模型
        print("正在初始化模型 (Embedding + LLM)...")
        self.text_embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        self.image_embed_model = ClipEmbedding(model_name="ViT-L/14")
        
        self.llm = OpenAILike(
            model="qwen3-vl-plus", 
            is_chat_model=True,
            is_function_calling_model=True
        )
        
        if self.eval_enabled:
            print("初始化评估模型 (DeepSeek)...")
            self.eval_llm = OpenAILike(
                model="DeepSeek-V3.2", 
                is_chat_model=True,
            )
        
        self.index = None
        self.query_engine = None

    def process_single_query(self, item):
        """
        Process a single query item.
        item: dict containing 'query', 'ground_truth' (optional), and other metadata
        """
        query_str = item.get("query")
        ground_truth = item.get("ground_truth")
        
        if not query_str:
            return {"error": "No query string provided"}
            
        try:
            # Query
            response = self.query_engine.query(query_str)
            prediction = str(response).strip()
            
            result = {
                "query": query_str,
                "prediction": prediction,
                "ground_truth": ground_truth
            }
            
            # Evaluate
            if self.eval_enabled and ground_truth:
                eval_result = self.evaluate_prediction(query_str, ground_truth, prediction)
                result["eval_result"] = eval_result
            
            return result
        except Exception as e:
            return {"query": query_str, "error": str(e)}

    def batch_query(self, query_items, max_workers=5):
        """
        Run batch queries in parallel.
        :param query_items: list of dicts, each with 'query' and optional 'ground_truth'
        :param max_workers: number of threads
        :return: list of result dicts
        """
        if not self.query_engine:
            raise Exception("Query Engine not initialized. Call build_index() first.")
            
        print(f"\nProcessing {len(query_items)} queries with {max_workers} workers...")
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map items to futures
            future_to_item = {executor.submit(self.process_single_query, item): item for item in query_items}
            
            # Process as they complete
            for future in tqdm(as_completed(future_to_item), total=len(query_items), desc="Querying"):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as e:
                    item = future_to_item[future]
                    results.append({"query": item.get("query"), "error": str(e)})
                    
        return results

    def _resolve_image_path(self, rel_path):
        """
        解析图片绝对路径
        """
        if not rel_path:
            return None
            
        rel_path_norm = rel_path.replace("\\", "/").replace("/", os.sep)
        
        candidates = [
            os.path.join("dataset", rel_path_norm),
            os.path.join("..", "dataset", rel_path_norm),
            os.path.join("dataset", os.path.dirname(rel_path_norm), rel_path_norm),
            os.path.join("..", "dataset", os.path.dirname(rel_path_norm), rel_path_norm)
        ]
        
        parts = rel_path_norm.split(os.sep)
        if len(parts) > 1:
            subdir = parts[0]
            rest = os.path.join(*parts[1:])
            candidates.append(os.path.join("dataset", subdir, subdir, rest))
            candidates.append(os.path.join("..", "dataset", subdir, subdir, rest))

        for p in candidates:
            if os.path.exists(p):
                return os.path.abspath(p)
        
        return None

    def evaluate_prediction(self, query, ground_truth, prediction):
        """
        使用裁判模型评估回答正确性
        """
        if not self.eval_enabled:
            return None

        prompt = (
            f"You are an impartial judge evaluating the correctness of an answer to a question.\n"
            f"Question: {query}\n"
            f"Ground Truth: {ground_truth}\n"
            f"Prediction: {prediction}\n\n"
            f"Does the prediction correctly answer the question based on the ground truth? "
            f"Focus on semantic meaning. "
            f"Respond with a JSON object containing 'correct' (boolean) and 'reason' (string)."
            f"Do not output markdown code blocks, just the raw JSON string."
        )

        messages = [
            ChatMessage(role="system", content="You are a helpful assistant that evaluates QA results."),
            ChatMessage(role="user", content=prompt)
        ]

        try:
            response = self.eval_llm.chat(messages)
            content = response.message.content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            
            result_json = json.loads(content.strip())
            return result_json
        except Exception as e:
            return {"correct": False, "reason": f"Evaluation failed: {str(e)}"}

    def query(self, query_str, ground_truth=None):
        """
        执行查询并(可选)评估
        """
        if not self.query_engine:
            raise Exception("Query Engine not initialized. Call build_index() first.")
        
        print(f"\n执行查询: {query_str}")
        response = self.query_engine.query(query_str)
        prediction = str(response).strip()
        
        eval_result = None
        if self.eval_enabled and ground_truth:
            print("正在评估回答...")
            eval_result = self.evaluate_prediction(query_str, ground_truth, prediction)
            if eval_result:
                print(f"评估结果: {'✅ 正确' if eval_result.get('correct') else '❌ 错误'} - {eval_result.get('reason')}")

        return response, eval_result

    def build_index(self, force_rebuild=False):
        raise NotImplementedError("Subclasses must implement build_index")


class DualModalPipeline(BasePipeline):
    """
    双模态 Pipeline:
    - ImageDocument: 包含图片和 text 字段
    - TextDocument: 独立的 Document，确保文本被索引
    """
    def __init__(self, dataset_path, storage_dir, eval_enabled=True, image_key="image", caption_key="original_caption"):
        super().__init__(dataset_path, storage_dir, eval_enabled)
        self.image_key = image_key
        self.caption_key = caption_key

    def build_index(self, force_rebuild=False):
        if not force_rebuild and os.path.exists(self.storage_dir):
            print(f"从 {self.storage_dir} 加载现有索引...")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            self.index = load_index_from_storage(
                storage_context,
                embed_model=self.text_embed_model,
                image_embed_model=self.image_embed_model
            )
        else:
            print(f"正在加载数据集并构建新索引 (DualModal: {self.image_key} + {self.caption_key})...")
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = []
            
            for entry in data:
                img_rel_path = entry.get(self.image_key)
                caption = entry.get(self.caption_key)
                
                # Skip if missing required data
                if not img_rel_path:
                    continue
                
                img_abs_path = self._resolve_image_path(img_rel_path)
                
                if not img_abs_path:
                    # Try resolving with default image path if edited path fails/is same structure? 
                    # Actually _resolve_image_path handles relative paths. 
                    # If edited image doesn't exist, we skip.
                    print(f"Warning: Image not found for {img_rel_path}")
                    continue

                # 1. ImageDocument (带 text)
                img_doc = ImageDocument(
                    image_path=img_abs_path,
                    text=caption if caption else "", 
                    metadata={
                        "file_name": os.path.basename(img_abs_path),
                        "type": "image_node",
                        "doc_id": img_rel_path,
                        "caption_type": self.caption_key,
                        "image_type": self.image_key
                    }
                )
                documents.append(img_doc)
                
                # 2. TextDocument (独立)
                if caption:
                    text_doc = Document(
                        text=caption,
                        metadata={
                            "related_image": os.path.basename(img_abs_path),
                            "type": "text_node",
                            "doc_id": img_rel_path,
                            "caption_type": self.caption_key,
                            "image_type": self.image_key
                        }
                    )
                    documents.append(text_doc)
            
            print(f"共创建 {len(documents)} 个文档.")
            print("开始构建 MultiModalVectorStoreIndex...")
            
            self.index = MultiModalVectorStoreIndex.from_documents(
                documents,
                embed_model=self.text_embed_model,
                image_embed_model=self.image_embed_model
            )
            
            if self.storage_dir:
                print(f"保存索引到 {self.storage_dir}...")
                self.index.storage_context.persist(persist_dir=self.storage_dir)

        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=1,
            image_similarity_top_k=1
        )
        print("DualModalPipeline 初始化完成。")


class ImagePipeline(BasePipeline):
    """
    纯图片 Pipeline (Baseline 风格):
    - 仅创建 ImageDocument (带 text)
    - 不创建独立的 TextDocument
    """
    def build_index(self, force_rebuild=False):
        if not force_rebuild and os.path.exists(self.storage_dir):
            print(f"从 {self.storage_dir} 加载现有索引...")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            self.index = load_index_from_storage(
                storage_context,
                embed_model=self.text_embed_model,
                image_embed_model=self.image_embed_model
            )
        else:
            print("正在加载数据集并构建新索引 (ImagePipeline: Only ImageDocument)...")
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = []
            
            for entry in data:
                img_rel_path = entry.get("image") or entry.get("edited_image")
                caption = entry.get("original_caption") or entry.get("fake_caption")
                
                img_abs_path = self._resolve_image_path(img_rel_path)
                
                if not img_abs_path:
                    print(f"Warning: Image not found for {img_rel_path}")
                    continue

                # 仅创建 ImageDocument
                img_doc = ImageDocument(
                    image_path=img_abs_path,
                    text=caption if caption else "",
                    metadata={
                        "file_name": os.path.basename(img_abs_path),
                        "type": "image_node",
                        "doc_id": img_rel_path
                    }
                )
                documents.append(img_doc)
            
            print(f"共创建 {len(documents)} 个文档.")
            print("开始构建 MultiModalVectorStoreIndex...")
            
            self.index = MultiModalVectorStoreIndex.from_documents(
                documents,
                embed_model=self.text_embed_model,
                image_embed_model=self.image_embed_model
            )
            
            if self.storage_dir:
                print(f"保存索引到 {self.storage_dir}...")
                self.index.storage_context.persist(persist_dir=self.storage_dir)

        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=1,
            image_similarity_top_k=1
        )
        print("ImagePipeline 初始化完成。")
