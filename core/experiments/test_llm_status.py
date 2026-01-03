import os
import sys
import dotenv
dotenv.load_dotenv()
# Add project root to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from core.experiments.pipeline import OpenAILike
from llama_index.core.base.llms.types import ChatMessage

def test_llm():
    try:
        print("Testing DeepSeek-V3.2...")
        llm = OpenAILike(model="DeepSeek-V3.2", is_chat_model=True)
        resp = llm.chat([ChatMessage(role="user", content="Say hello")])
        print(f"DeepSeek Response: {resp.message.content}")
        
        print("Testing qwen3-vl-plus...")
        llm2 = OpenAILike(model="qwen3-vl-plus", is_chat_model=True)
        resp2 = llm2.chat([ChatMessage(role="user", content="Say hi")])
        print(f"Qwen Response: {resp2.message.content}")
        
        print("LLMs are working.")
        return True
    except Exception as e:
        print(f"LLM Failed: {e}")
        return False

if __name__ == "__main__":
    test_llm()
