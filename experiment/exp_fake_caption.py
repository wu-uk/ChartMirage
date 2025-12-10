import json
import os
import concurrent.futures
from datetime import datetime
from dotenv import load_dotenv 
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.schema import ImageDocument
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Setup Models
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not set.")
    exit(1)

# Setup logging directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"exp_fake_caption_{timestamp}.jsonl")
print(f"Logging results to: {log_file}")

print("Initializing models...")
# Use OpenAI Embedding for Text (Captions) to handle long text
text_embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# Use CLIP Embedding for Images
image_embed_model = ClipEmbedding(model_name="ViT-L/14")

llm = OpenAILike(
    model_name="qwen3-vl-plus", 
    is_chat_model=True,
    is_function_calling_model=True
)

# Initialize separate LLM for evaluation (Judge)
eval_llm = OpenAILike(
    model_name="DeepSeek-V3.2", 
    is_chat_model=True,
    api_key=os.environ.get("OPENAI_API_KEY"),
    api_base=os.environ.get("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)

# 2. Load Dataset
dataset_path = os.path.join("dataset", "final_qa_merged_unified.json")
if not os.path.exists(dataset_path):
    # Try looking one level up if running from experiment folder
    dataset_path = os.path.join("..", "dataset", "final_qa_merged_unified.json")

print(f"Loading dataset from: {dataset_path}")
with open(dataset_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3. Create Documents (Images + FAKE Captions)
documents = []
image_map = {} # To avoid duplicate images

print("Creating documents (using Fake Captions)...")
# Iterate through data to gather unique images and their captions
for entry in data:
    img_rel_path = entry["image"] # e.g. "images_merged/img_000001.jpg"
    
    # Resolve path
    if os.path.exists(os.path.join("dataset", img_rel_path)):
        img_abs_path = os.path.abspath(os.path.join("dataset", img_rel_path))
    elif os.path.exists(os.path.join("..", "dataset", img_rel_path)):
        img_abs_path = os.path.abspath(os.path.join("..", "dataset", img_rel_path))
    else:
        print(f"Warning: Image not found: {img_rel_path}")
        continue
    
    if img_abs_path not in image_map:
        doc = ImageDocument(
            image_path=img_abs_path,
            text=entry["fake_caption"], # Using FAKE caption
            metadata={"file_name": os.path.basename(img_abs_path)}
        )
        image_map[img_abs_path] = doc
        documents.append(doc)

print(f"Loaded {len(documents)} unique documents.")


STORAGE_DIR = "storage_fake_caption"

# 4. Build or Load Index
if os.path.exists(STORAGE_DIR):
    print(f"Loading existing index from {STORAGE_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(
        storage_context,
        embed_model=text_embed_model,
        image_embed_model=image_embed_model
    )
else:
    print("Building Multi-modal Index...")
    index = MultiModalVectorStoreIndex.from_documents(
        documents,
        embed_model=text_embed_model,
        image_embed_model=image_embed_model,
    )
    # Persist index to disk
    print(f"Saving index to {STORAGE_DIR}...")
    index.storage_context.persist(persist_dir=STORAGE_DIR)

# 5. Run Experiment (Query -> Answer)
retriever_engine = index.as_query_engine(
    llm=llm,
    similarity_top_k=1,
    image_similarity_top_k=1
)

def evaluate_prediction(query, ground_truth, prediction):
    """
    Evaluates if the prediction matches the ground truth using the shared LLM.
    """
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
        response = eval_llm.chat(messages)
        content = response.message.content.strip()
        # Clean up code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        result_json = json.loads(content.strip())
        return result_json
    except Exception as e:
        # print(f"Error evaluating: {e}") # Reduce noise in threads
        return {"correct": False, "reason": f"Evaluation failed: {str(e)}"}

def process_query(entry, idx):
    """
    Process a single query entry.
    """
    query = entry["query"]
    target_img_name = os.path.basename(entry["image"])
    ground_truth = entry["answer"]
    
    try:
        response = retriever_engine.query(query)
        
        # Check Retrieval
        retrieved_img = "None"
        score = 0.0
        if response.source_nodes:
            node = response.source_nodes[0]
            retrieved_img = node.metadata.get("file_name", "Unknown")
            score = node.score
        
        is_retrieval_correct = (retrieved_img == target_img_name)
        prediction = str(response).strip()
        
        # Evaluate prediction
        eval_result = evaluate_prediction(query, ground_truth, prediction)

        result = {
            "id": idx,
            "query": query,
            "target_image": target_img_name,
            "retrieved_image": retrieved_img,
            "retrieval_score": score,
            "retrieval_success": is_retrieval_correct,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "evaluation": eval_result,
            "timestamp": datetime.now().isoformat()
        }
        
        # Print short status to console
        status = "HIT " if is_retrieval_correct else "MISS"
        eval_status = "CORRECT" if eval_result.get("correct") else "WRONG  "
        print(f"[{idx}] {status} | Eval: {eval_status} | Target: {target_img_name}")
        
        return result

    except Exception as e:
        print(f"Error processing query {idx}: {e}")
        return {
            "id": idx,
            "query": query,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

print("\n--- Starting Experiment: Image + Fake Caption (Multi-threaded) ---")

# Use ThreadPoolExecutor for concurrent processing
# Adjust max_workers based on your API rate limits
results = []
max_workers = 8
limit_entries = len(data) # Full run

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit tasks
    future_to_entry = {
        executor.submit(process_query, entry, i): i 
        for i, entry in enumerate(data[:limit_entries])
    }
    
    # Collect results as they complete
    with open(log_file, "w", encoding="utf-8") as f:
        for future in concurrent.futures.as_completed(future_to_entry):
            res = future.result()
            results.append(res)
            # Write to log file immediately (JSONL format)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()

print(f"\nExperiment completed. Results saved to {log_file}")
