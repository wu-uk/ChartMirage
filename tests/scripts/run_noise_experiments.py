import os
import json
import sys
import glob
from llama_index.core import Settings
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import ImageDocument
from llama_index.core.indices import MultiModalVectorStoreIndex

# Add project root to sys.path
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

from core.experiments.pipeline import DualModalPipeline

MAX_WORKERS = 10
RESULT_DIR = os.path.join(base_dir, "outputs/results")
DATASET_PATH = os.path.join(base_dir, "dataset/final_qa_merged_unified.json")
NOISE_BASE_DIR = os.path.join(base_dir, "dataset/images_noise_fake")

class NoisePipeline(DualModalPipeline):
    def __init__(self, dataset_path, storage_dir, noise_dir, eval_enabled=True, caption_key="original_caption"):
        super().__init__(dataset_path, storage_dir, eval_enabled, image_key="edited_image", caption_key=caption_key)
        self.noise_dir = noise_dir

    def build_index(self, force_rebuild=False):
        if not force_rebuild and os.path.exists(self.storage_dir) and os.listdir(self.storage_dir):
            print(f"Loading existing index from {self.storage_dir}...")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            self.index = load_index_from_storage(
                storage_context,
                embed_model=self.text_embed_model,
                image_embed_model=self.image_embed_model
            )
        else:
            print(f"Building new index from noise dir: {self.noise_dir}")
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            documents = []
            valid_count = 0
            missing_count = 0

            for entry in data:
                # Get filename from edited_image path
                rel_path = entry.get("edited_image") # e.g. "images_merged_fake/0.png"
                if not rel_path: 
                    continue
                
                filename = os.path.basename(rel_path)
                
                # Construct absolute path to noisy image
                img_abs_path = os.path.join(self.noise_dir, filename)
                
                if not os.path.exists(img_abs_path):
                    missing_count += 1
                    # print(f"Warning: Noisy image not found: {img_abs_path}")
                    continue

                caption = entry.get(self.caption_key)

                # Create documents
                img_doc = ImageDocument(
                    image_path=img_abs_path,
                    text=caption if caption else "",
                    metadata={
                        "file_name": filename,
                        "type": "image_node",
                        "doc_id": rel_path,
                        "caption_type": self.caption_key,
                        "image_type": "noise_image"
                    }
                )
                documents.append(img_doc)

                if caption:
                    text_doc = Document(
                        text=caption,
                        metadata={
                            "related_image": filename,
                            "type": "text_node",
                            "doc_id": rel_path,
                            "caption_type": self.caption_key,
                            "image_type": "noise_image"
                        }
                    )
                    documents.append(text_doc)
                valid_count += 1
            
            print(f"Documents created: {len(documents)} (Images found: {valid_count}, Missing: {missing_count})")
            if not documents:
                print("No documents created. Aborting index build.")
                return

            print("Indexing...")
            self.index = MultiModalVectorStoreIndex.from_documents(
                documents,
                embed_model=self.text_embed_model,
                image_embed_model=self.image_embed_model
            )
            
            if self.storage_dir:
                print(f"Saving index to {self.storage_dir}...")
                self.index.storage_context.persist(persist_dir=self.storage_dir)
        
        # Initialize Query Engine
        self.query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=1,
            image_similarity_top_k=1
        )

def run_experiment(noise_type, limit=10000):
    exp_name = f"Noise_{noise_type}"
    print(f"\n{'='*50}")
    print(f"Running Experiment: {exp_name}")
    print(f"{'='*50}")
    
    storage_dir = os.path.join(base_dir, f"storage/storage_{exp_name}")
    noise_dir = os.path.join(NOISE_BASE_DIR, noise_type)
    
    if not os.path.exists(noise_dir):
        print(f"Error: Noise directory {noise_dir} does not exist.")
        return

    # Initialize Pipeline
    pipeline = NoisePipeline(
        dataset_path=DATASET_PATH,
        storage_dir=storage_dir,
        noise_dir=noise_dir,
        eval_enabled=True,
        caption_key="original_caption"
    )
    
    # Build Index
    try:
        pipeline.build_index()
    except Exception as e:
        print(f"Index build failed: {e}")
        return

    # Load Queries
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    query_items = []
    count = 0
    for entry in data:
        if count >= limit:
            break
            
        query_str = entry.get("query")
        answer = entry.get("answer")
        fake_answer = entry.get("fake_answer")
        
        if answer == fake_answer:
            continue
            
        item = {
            "query": query_str,
            "ground_truth": answer,
            "fake_answer": fake_answer,
            "id": count
        }
        query_items.append(item)
        count += 1
    
    # Execute Batch
    results = pipeline.batch_query(query_items, max_workers=MAX_WORKERS)
    
    # Save Results
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
        
    output_file = f"{RESULT_DIR}/results_{exp_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    noise_types = ["gaussian_noise", "salt_pepper_noise", "gaussian_blur", "rotation"]
    
    for nt in noise_types:
        run_experiment(nt)
