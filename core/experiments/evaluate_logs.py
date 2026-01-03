import json
import os
import argparse
import concurrent.futures
from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.llms import ChatMessage

# Load environment variables
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not set.")
    exit(1)

def evaluate_prediction(llm, query, ground_truth, prediction):
    """
    Evaluates if the prediction matches the ground truth using an LLM.
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
        response = llm.chat(messages)
        content = response.message.content.strip()
        # Clean up code blocks if present
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        
        result_json = json.loads(content.strip())
        return result_json
    except Exception as e:
        # print(f"Error evaluating: {e}")
        return {"correct": False, "reason": f"Evaluation failed: {str(e)}"}

def process_entry(entry_data, llm):
    if isinstance(entry_data, str):
        if not entry_data.strip():
            return None
        try:
            entry = json.loads(entry_data)
        except json.JSONDecodeError:
            return None
    else:
        entry = entry_data

    # Skip if already evaluated
    if "evaluation" in entry:
        return entry

    query = entry.get("query", "")
    ground_truth = entry.get("ground_truth", "")
    prediction = entry.get("prediction", "")

    # Perform evaluation
    eval_result = evaluate_prediction(llm, query, ground_truth, prediction)
    
    entry["evaluation"] = eval_result
    return entry

def main():
    parser = argparse.ArgumentParser(description="Evaluate QA logs using LLM.")
    parser.add_argument("log_file", help="Path to the JSON/JSONL log file to evaluate")
    parser.add_argument("--workers", type=int, default=8, help="Number of threads")
    args = parser.parse_args()

    log_file = args.log_file
    if not os.path.exists(log_file):
        print(f"File not found: {log_file}")
        return

    output_file = log_file.replace(".json", "_evaluated.json").replace(".jsonl", "")
    # Fix extension if it was just .json
    if not output_file.endswith(".json") and not output_file.endswith(".jsonl"):
         output_file += ".jsonl" # default to jsonl output for evaluated
    
    # If input was .json but output became _evaluated.json, let's make it .jsonl for streaming write safety
    # Or keep it consistent. The original script wrote line by line, so JSONL output is safer.
    if output_file.endswith(".json"):
        output_file = output_file.replace(".json", ".jsonl")

    print(f"Evaluating logs from: {log_file}")
    print(f"Saving results to: {output_file}")
    print(f"Using model: DeepSeek-V3.2")

    # Initialize LLM
    llm = OpenAILike(
        model="DeepSeek-V3.2", 
        is_chat_model=True,
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_base=os.environ.get("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    )

    entries = []
    # Try to load as JSON Array first, then fall back to JSONL
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content.startswith("[") and content.endswith("]"):
                entries = json.loads(content)
                print("Detected JSON Array format.")
            else:
                # Treat as JSONL
                entries = content.splitlines()
                print("Detected JSONL format.")
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    total_lines = len(entries)
    evaluated_count = 0
    correct_count = 0

    print(f"Processing {total_lines} entries with {args.workers} threads...")

    with open(output_file, "w", encoding="utf-8") as fout:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_entry = {executor.submit(process_entry, entry, llm): entry for entry in entries}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_entry)):
                try:
                    result = future.result()
                    if result:
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                        
                        # Stats
                        if "evaluation" in result:
                            evaluated_count += 1
                            if result["evaluation"].get("correct"):
                                correct_count += 1
                            
                            status = "CORRECT" if result["evaluation"].get("correct") else "WRONG  "
                            print(f"[{evaluated_count}/{total_lines}] {status} | Query: {result.get('query')[:50]}...")
                except Exception as e:
                    print(f"Error processing entry: {e}")

    print(f"\nEvaluation complete.")
    print(f"Total Evaluated: {evaluated_count}")
    print(f"Total Correct: {correct_count}")
    if evaluated_count > 0:
        print(f"Accuracy: {correct_count / evaluated_count * 100:.2f}%")

if __name__ == "__main__":
    main()
