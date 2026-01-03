import os
import sys
import json
import random
import argparse
import copy
from datetime import datetime
from dotenv import load_dotenv
import qdrant_client
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, ImageNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Try to find project root by looking for 'dataset' directory
if os.path.exists("dataset"):
    base_dir = "."
elif os.path.exists("../../dataset"):
    base_dir = "../.."
else:
    base_dir = ".."

from core.experiments.consistency_pipeline import ConsistencyPipeline
from core.experiments.pipeline import OpenAILike

from llama_index.core.vector_stores import VectorStoreQuery

class MixedRetriever(BaseRetriever):
    def __init__(self, text_embed_model, image_embed_model, 
                 real_text_store, real_image_store, 
                 fake_text_store, fake_image_store, 
                 allowed_ids, top_k=3):
        super().__init__()
        self.text_embed_model = text_embed_model
        self.image_embed_model = image_embed_model
        self.real_text_store = real_text_store
        self.real_image_store = real_image_store
        self.fake_text_store = fake_text_store
        self.fake_image_store = fake_image_store
        self.allowed_ids = allowed_ids
        self.top_k = top_k

    def _query_store(self, store, query_embedding, mode="text"):
        query_obj = VectorStoreQuery(
            query_embedding=query_embedding, 
            similarity_top_k=self.top_k,
            mode="default"
        )
        result = store.query(query_obj)
        nodes = []
        if result.nodes:
            for i, node in enumerate(result.nodes):
                score = result.similarities[i] if result.similarities else 0.0
                nodes.append(NodeWithScore(node=node, score=score))
        return nodes

    def _retrieve(self, query_bundle):
        query_str = query_bundle.query_str
        
        # 1. Generate Embeddings
        # Text Query Embedding (for Text Store)
        text_query_emb = self.text_embed_model.get_query_embedding(query_str)
        
        # Image Query Embedding (for Image Store - using CLIP text encoder)
        image_query_emb = self.image_embed_model.get_query_embedding(query_str)
        
        # 2. Query All Stores
        # Real
        nodes_real_text = self._query_store(self.real_text_store, text_query_emb)
        nodes_real_image = self._query_store(self.real_image_store, image_query_emb)
        
        # Fake
        nodes_fake_text = self._query_store(self.fake_text_store, text_query_emb)
        nodes_fake_image = self._query_store(self.fake_image_store, image_query_emb)
        
        # 3. Merge and Filter
        all_candidates = nodes_real_text + nodes_real_image + nodes_fake_text + nodes_fake_image
        
        combined = []
        seen_ids = set()
        
        for node_score in all_candidates:
            node = node_score.node
            doc_id = node.metadata.get("doc_id")
            
            if doc_id in self.allowed_ids:
                if node.node_id not in seen_ids:
                    combined.append(node_score)
                    seen_ids.add(node.node_id)
        
        # Sort by score
        # combined.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        
        # Separate Text and Image for Top-K selection to ensure modality coverage
        c_text = [n for n in combined if not isinstance(n.node, ImageNode)]
        c_image = [n for n in combined if isinstance(n.node, ImageNode)]
        
        c_text.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        c_image.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        
        # Return top-k from each modality
        return c_text[:self.top_k] + c_image[:self.top_k]

from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock

class SimulatedConsistencyPipeline(ConsistencyPipeline):
    def __init__(self, dataset_path, real_storage_dir, fake_storage_dir, allowed_ids):
        # Skip QdrantPipeline.__init__ index loading to avoid embedding
        # Manually init necessary components
        load_dotenv()
        
        self.dataset_path = dataset_path
        self.eval_enabled = True
        self.top_k = 3
        
        # Bypass query_engine check in batch_query
        self.query_engine = True
        
        # Initialize LLMs
        self.llm = OpenAILike(model="qwen3-vl-plus", is_chat_model=True)
        self.eval_llm = OpenAILike(model="DeepSeek-V3.2", is_chat_model=True)
        
        # Initialize Embeddings (Must match what was used to build the index)
        self.text_embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        self.image_embed_model = ClipEmbedding(model_name="ViT-L/14")
        
        # Set Global Settings to ensure correct embedding is used during retrieval
        Settings.embed_model = self.text_embed_model
        Settings.llm = self.llm
        
        # DEBUG: Check dimensions
        try:
            print(f"DEBUG: Embed Model: {Settings.embed_model.model_name}")
            test_embed = Settings.embed_model.get_text_embedding("test")
            print(f"DEBUG: Embedding Dimension: {len(test_embed)}")
        except Exception as e:
            print(f"DEBUG: Failed to check embedding: {e}")

        # Load Real Index
        print(f"Loading Real Index from {real_storage_dir}...")
        self.real_client = qdrant_client.QdrantClient(path=os.path.join(real_storage_dir, "qdrant_db"))
        real_text_store = QdrantVectorStore(client=self.real_client, collection_name="text_collection")
        real_image_store = QdrantVectorStore(client=self.real_client, collection_name="image_collection")
        
        # DEBUG: Check collection info
        try:
            print(f"DEBUG: Real Text Collection Count: {self.real_client.count(collection_name='text_collection').count}")
            print(f"DEBUG: Real Image Collection Count: {self.real_client.count(collection_name='image_collection').count}")
            # Check vector size?
            # Qdrant client doesn't easily expose vector size in count, but we can try getting info
            print(f"DEBUG: Real Text Info: {self.real_client.get_collection('text_collection').config.params.vectors}")
            print(f"DEBUG: Real Image Info: {self.real_client.get_collection('image_collection').config.params.vectors}")
        except Exception as e:
            print(f"DEBUG: Failed to get collection info: {e}")

        real_storage_ctx = StorageContext.from_defaults(
            persist_dir=real_storage_dir,
            vector_store=real_text_store,
            image_store=real_image_store
        )
        
        self.real_index = load_index_from_storage(
            real_storage_ctx,
            embed_model=self.text_embed_model,
            image_embed_model=self.image_embed_model
        )

        # Load Fake Index
        print(f"Loading Fake Index from {fake_storage_dir}...")
        self.fake_client = qdrant_client.QdrantClient(path=os.path.join(fake_storage_dir, "qdrant_db"))
        fake_text_store = QdrantVectorStore(client=self.fake_client, collection_name="text_collection")
        fake_image_store = QdrantVectorStore(client=self.fake_client, collection_name="image_collection")
        
        fake_storage_ctx = StorageContext.from_defaults(
            persist_dir=fake_storage_dir,
            vector_store=fake_text_store,
            image_store=fake_image_store
        )
        
        self.fake_index = load_index_from_storage(
            fake_storage_ctx,
            embed_model=self.text_embed_model,
            image_embed_model=self.image_embed_model
        )
        
        # Create Mixed Retriever
        self.retriever = MixedRetriever(
            text_embed_model=self.text_embed_model,
            image_embed_model=self.image_embed_model,
            real_text_store=real_text_store,
            real_image_store=real_image_store,
            fake_text_store=fake_text_store,
            fake_image_store=fake_image_store,
            allowed_ids=allowed_ids,
            top_k=self.top_k
        )

    def close(self):
        if hasattr(self, 'real_client'):
            self.real_client.close()
        if hasattr(self, 'fake_client'):
            self.fake_client.close()
            
    def generate_answer(self, query, text_nodes, image_nodes):
        """
        Generate answer using retrieved nodes.
        """
        context_str = "\n".join([n.get_content() for n in text_nodes])
        
        # Prepare blocks
        blocks = [
            TextBlock(text=f"Context information is below.\n---------------------\n{context_str}\n---------------------\nGiven the context information and the image, answer the query.\nQuery: {query}")
        ]
        
        # Add images
        for node in image_nodes:
            image_path = None
            if hasattr(node, "image_path"):
                image_path = node.image_path
            elif "file_path" in node.metadata:
                image_path = node.metadata["file_path"]
            
            if image_path and os.path.exists(image_path):
                blocks.append(ImageBlock(path=image_path))
                # Just use the first valid image for now, as most VLMs handle one image best or we limit confusion
                # Qwen-VL supports multiple, but let's stick to retrieval logic
                # ConsistencyPipeline.check_consistency used image_nodes[0]
                # Let's use all retrieved images?
                # For safety/simplicity, let's use the first one if multiple, or all.
                # Qwen supports interleaved.
                pass
        
        # Construct message
        message = ChatMessage(role="user", blocks=blocks)
        
        response = self.llm.chat([message])
        return response.message.content

    def process_single_query(self, item):
        query = item.get("query")
        ground_truth = item.get("ground_truth")
        fake_answer = item.get("fake_answer")

        if not query:
            return {"error": "No query provided"}

        # 1. Retrieve (using MixedRetriever)
        nodes = self.retriever.retrieve(query)
        
        # Separate Text and Image Nodes
        text_nodes = [n.node for n in nodes if not isinstance(n.node, ImageNode)]
        image_nodes = [n.node for n in nodes if isinstance(n.node, ImageNode)]
        
        # 2. Consistency Check
        is_consistent, reason = self.check_consistency(query, text_nodes, image_nodes)

        result = {
            "query": query,
            "prediction": None,
            "ground_truth": ground_truth,
            "fake_answer": fake_answer,
            "is_refusal": False,
            "consistency_check": {
                "consistent": is_consistent,
                "reason": reason
            },
            "retrieved_nodes": [n.get_content()[:100]+"..." for n in text_nodes]
        }

        if not is_consistent:
            result["is_refusal"] = True
            result["prediction"] = f"Refused to answer: Inconsistent information detected. ({reason})"
            result["asr_result"] = {"match": False, "reason": "Refused"}
            return result

        # 3. Generate Answer (if consistent)
        response = self.generate_answer(query, text_nodes, image_nodes)
        result["prediction"] = str(response)

        # 4. Evaluate
        if self.eval_enabled:
            eval_res = self.evaluate_prediction(query, result["prediction"], ground_truth)
            result["eval_result"] = eval_res
            
            asr_match = self.check_attack_success(query, fake_answer, result["prediction"])
            result["asr_result"] = {"match": asr_match}

        return result

def get_allowed_ids_and_dataset(original_path, ratio, seed=42):
    with open(original_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    random.seed(seed)
    indices = list(range(len(data)))
    random.shuffle(indices)
    
    num_poison = int(len(data) * ratio)
    poison_indices = set(indices[:num_poison])
    
    allowed_ids = set()
    query_items = []
    
    poisoned_count = 0
    
    for i, entry in enumerate(data):
        # Always add Clean ID (Real Image)
        if entry.get("image"):
            allowed_ids.add(entry["image"])

        is_poisoned = False
        if i in poison_indices:
            # Poisoned: Add Fake Image ID if available
            if entry.get("edited_image"):
                allowed_ids.add(entry["edited_image"])
                is_poisoned = True
                poisoned_count += 1
        
        query_items.append({
            "query": entry["query"],
            "ground_truth": entry.get("answer"),
            "fake_answer": entry.get("fake_answer"),
            "is_poisoned": is_poisoned
        })
            
    return allowed_ids, query_items, poisoned_count

def run_simulated_ablation(ratio, args):
    print(f"\n{'='*60}")
    print(f"Starting Simulated Ablation: {ratio*100}% Poisoned Images")
    print(f"{'='*60}")

    dataset_source = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
    real_storage = os.path.join(base_dir, "storage/storage_consistency_exp_1_baseline")
    fake_storage = os.path.join(base_dir, "storage/storage_consistency_exp_3_image_poison")
    
    # 1. Prepare Allowed IDs
    allowed_ids, query_items, p_count = get_allowed_ids_and_dataset(dataset_source, ratio)

    print(f"Simulating: {p_count}/{len(query_items)} poisoned items.")
    
    if args.limit:
        print(f"Limiting to {args.limit} queries.")
        query_items = query_items[:args.limit]
        # Note: allowed_ids remains full set, which is fine (retriever will just not be called for skipped queries)

    # 2. Init Pipeline
    pipeline = SimulatedConsistencyPipeline(
        dataset_path=dataset_source,
        real_storage_dir=real_storage,
        fake_storage_dir=fake_storage,
        allowed_ids=allowed_ids
    )

    # 3. Batch Query
    print(f"Running queries...")
    results = pipeline.batch_query(query_items, max_workers=args.workers)
    
    # Close pipeline to release Qdrant locks
    pipeline.close()

    # 4. Analyze
    # Map back is_poisoned status
    # Create a unique key map: query + gt? Or just rely on order if single threaded?
    # Batch query returns results. Order is NOT guaranteed.
    # We must match.
    # Let's build a map from query_str to is_poisoned.
    # Warning: Duplicates exist.
    
    # Improved Matching:
    # Use a queue for duplicates
    q_map = {}
    for item in query_items:
        q = item["query"]
        if q not in q_map:
            q_map[q] = []
        q_map[q].append(item["is_poisoned"])
    
    q_consumption = {k: 0 for k in q_map}

    stats = {
        "total": 0,
        "clean": {"total": 0, "refusals": 0, "correct": 0, "asr": 0},
        "poisoned": {"total": 0, "refusals": 0, "correct": 0, "asr": 0}
    }
    
    for res in results:
        q = res.get("query")
        if not q: continue
        
        # Determine subset
        is_poisoned = False
        if q in q_map:
            idx = q_consumption[q]
            if idx < len(q_map[q]):
                is_poisoned = q_map[q][idx]
                q_consumption[q] += 1
        
        subset = stats["poisoned"] if is_poisoned else stats["clean"]
        stats["total"] += 1
        subset["total"] += 1
        
        if res.get("is_refusal"):
            subset["refusals"] += 1
        else:
            if res.get("eval_result", {}).get("correct"):
                subset["correct"] += 1
            if res.get("asr_result", {}).get("match"):
                subset["asr"] += 1
                
    # Save Results
    out_file = os.path.join(base_dir, f"outputs/results/result_ablation_simulated_{int(ratio*100)}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"ratio": ratio, "stats": stats, "results": results}, f, indent=2, ensure_ascii=False)
        
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    ratios = [0.25, 0.50, 0.75]
    table = []
    
    for r in ratios:
        try:
            stats = run_simulated_ablation(r, args)
            
            # Calculate metrics
            def calc(s):
                if s["total"] == 0: return 0, 0, 0
                return s["refusals"]/s["total"]*100, s["correct"]/s["total"]*100, s["asr"]/s["total"]*100
            
            c_ref, c_acc, c_asr = calc(stats["clean"])
            p_ref, p_acc, p_asr = calc(stats["poisoned"])
            
            # Overall
            total = stats["total"]
            all_ref = stats["clean"]["refusals"] + stats["poisoned"]["refusals"]
            all_acc = stats["clean"]["correct"] + stats["poisoned"]["correct"]
            all_asr = stats["clean"]["asr"] + stats["poisoned"]["asr"]
            
            table.append({
                "Ratio": f"{r*100:.0f}%",
                "Ov_Ref": f"{all_ref/total*100:.1f}%",
                "Ov_Acc": f"{all_acc/total*100:.1f}%",
                "Ov_ASR": f"{all_asr/total*100:.1f}%",
                "Clean_Ref": f"{c_ref:.1f}%",
                "Pois_Ref": f"{p_ref:.1f}%",
                "Pois_ASR": f"{p_asr:.1f}%"
            })
            
        except Exception as e:
            print(f"Error at {r}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\n" + "="*110)
    print(f"{'Ratio':<8} | {'Ov Ref':<8} | {'Ov Acc':<8} | {'Ov ASR':<8} | {'Clean Ref':<10} | {'Pois Ref':<10} | {'Pois ASR':<10}")
    print("-" * 110)
    for row in table:
        print(f"{row['Ratio']:<8} | {row['Ov_Ref']:<8} | {row['Ov_Acc']:<8} | {row['Ov_ASR']:<8} | {row['Clean_Ref']:<10} | {row['Pois_Ref']:<10} | {row['Pois_ASR']:<10}")
    print("="*110)

if __name__ == "__main__":
    main()
