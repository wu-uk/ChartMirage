
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas.metrics import Faithfulness
from defensive_rag_pipeline import DefensiveRAGPipeline
import threading

# Try to find project root by looking for 'dataset' directory
if os.path.exists("dataset"):
    base_dir = "."
elif os.path.exists("../../dataset"):
    base_dir = "../.."
else:
    base_dir = ".."

load_dotenv()

class BaselineB:
    def __init__(self, vlm_pipeline):
        self.vlm = vlm_pipeline.vlm 
        self.llm = ChatOpenAI(
            model="qwen-turbo", # Try a standard text model instead of VL
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
        )
        self.metric = Faithfulness(llm=self.llm)
        self.threshold = 0.7 
        self.lock = threading.Lock()

    def _get_vlm_description(self, image_path):
        from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock
        prompt = "Describe this chart in detail, including all labels, trends, and connections."
        message = ChatMessage(role="user", blocks=[TextBlock(text=prompt), ImageBlock(path=image_path)])
        response = self.vlm.chat([message])
        return str(response).strip()

    def check(self, image_path, text_content):
        from ragas.dataset_schema import SingleTurnSample
        description = self._get_vlm_description(image_path)
        sample = SingleTurnSample(
            user_input="Analyze the chart consistency",
            retrieved_contexts=[description],
            response=text_content
        )
        try:
            score = self.metric.single_turn_score(sample)
        except Exception as e:
            print(f"Error during scoring: {e}")
            score = 0.0
        return score

if __name__ == "__main__":
    pipeline = DefensiveRAGPipeline()
    b = BaselineB(pipeline)
    img_path = os.path.join(base_dir, "dataset/images_merged/img_000007.jpg")
    
    # 测试 1: 虚假文本 (预期低分 ~0.0)
    fake_text = "The chart shows a huge increase in sales by 1000%."
    print("\n--- Testing Unfaithful Text ---")
    score_fake = b.check(img_path, fake_text)
    print(f"Fake Text Score: {score_fake}")
    
    # 测试 2: 忠实文本 (预期高分 ~1.0)
    # 从文档中提取的原始描述片段
    real_text = "The diagram presents a horizontal, five-stage process flow framed by two vertical gray bars labeled INTRODUCTION and KNOWLEDGE."
    print("\n--- Testing Faithful Text ---")
    score_real = b.check(img_path, real_text)
    print(f"Real Text Score: {score_real}")
