import os
from dotenv import load_dotenv
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

# Load env
load_dotenv()

try:
    print("Initializing CLIP Embedding...")
    # This might trigger a download if not cached
    image_embed_model = ClipEmbedding(model_name="ViT-L/14")
    print("CLIP Initialized.")
except Exception as e:
    print(f"CLIP Failed: {e}")

try:
    print("Initializing OpenAI Embedding...")
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing!")
    else:
        text_embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        # Try a dummy embedding
        emb = text_embed_model.get_text_embedding("Hello world")
        print(f"OpenAI Embedding Success. Dim: {len(emb)}")
except Exception as e:
    print(f"OpenAI Failed: {e}")
