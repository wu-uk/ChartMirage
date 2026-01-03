import os
import sys
import json
import hashlib

# Add project root to sys.path for direct script execution
if __name__ == "__main__" or __name__.startswith("core."):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import torch
import cv2
import threading
import numpy as np
import clip
from PIL import Image
from dotenv import load_dotenv
from llama_index.core.base.llms.types import ChatMessage, ImageBlock, TextBlock
from core.experiments.pipeline import OpenAILike

# Fix for PaddleOCR missing CUDA libraries in this environment
import ctypes
import subprocess

nvidia_lib_path = '/home/ASC26team2/miniconda3/envs/ChartMirage/lib/python3.12/site-packages/nvidia/cu13/lib'
if os.path.exists(nvidia_lib_path):
    if nvidia_lib_path not in os.environ.get('LD_LIBRARY_PATH', ''):
        os.environ['LD_LIBRARY_PATH'] = nvidia_lib_path + ':' + os.environ.get('LD_LIBRARY_PATH', '')
        # We need to re-exec to pick up LD_LIBRARY_PATH for the dynamic linker
        try:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass

import torch.nn as nn
from paddleocr import PaddleOCR

class ConsistencyClassifier(nn.Module):
    def __init__(self, clip_model, dropout=0.3):
        super(ConsistencyClassifier, self).__init__()
        self.clip_model = clip_model
        self.embed_dim = clip_model.visual.output_dim 
        
        # Frequency Feature Processor
        self.freq_encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + 128 + 2, 512), 
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image, text, ocr_score, freq_feat):
        image_features = self.clip_model.encode_image(image).float()
        text_features = self.clip_model.encode_text(text).float()
        
        image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-8)
        text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-8)

        freq_embed = self.freq_encoder(freq_feat)
        ocr_score = ocr_score.float()
        combined = torch.cat((image_features, text_features, freq_embed, ocr_score), dim=1)
        return self.classifier(combined).squeeze()

class DefensiveRAGPipeline:
    def __init__(self, 
                 hash_registry_path=None,
                 level2_model_path=None,
                 vlm_model_name="qwen3-vl-plus"):
        
        load_dotenv()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Try to find project root by looking for 'dataset' directory
        if os.path.exists("dataset"):
            base_dir = "."
        elif os.path.exists("../dataset"):
            base_dir = ".."
        else:
            base_dir = "../.."

        if hash_registry_path is None:
            hash_registry_path = os.path.join(base_dir, "dataset/hash_registry.json")
        if level2_model_path is None:
            level2_model_path = os.path.join(base_dir, "core/experiments/consistency_check/best_model.pth")
        
        # Level 1: Hash-based Integrity

        if os.path.exists(hash_registry_path):
            with open(hash_registry_path, 'r') as f:
                self.hash_registry = json.load(f)
        else:
            self.hash_registry = {}
            print(f"Warning: Hash registry not found at {hash_registry_path}")

        # Level 2: Signal-level Check (Lightweight NN)
        print("Loading Level 2 model...")
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.l2_model = ConsistencyClassifier(self.clip_model).to(self.device)
        if os.path.exists(level2_model_path):
            state_dict = torch.load(level2_model_path, map_location=self.device)
            self.l2_model.load_state_dict(state_dict, strict=False)
            self.l2_model.eval()
        else:
            print(f"Warning: Level 2 model not found at {level2_model_path}")

        # Level 3: VLM-based Logic Auditor
        print("Initializing Level 3 VLM...")
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE")
        self.vlm = OpenAILike(
            model=vlm_model_name, 
            api_key=api_key,
            api_base=api_base,
            is_chat_model=True
        )
        
        # OCR for Level 2
        print("Initializing OCR for Level 2...")
        self.ocr_reader = PaddleOCR(use_textline_orientation=True, lang='en')
        
        # Thread safety lock for local models (CLIP, OCR, NN)
        self.gpu_lock = threading.Lock()
        
    def _calculate_hash(self, image_path, text_content):
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        hasher = hashlib.sha256()
        hasher.update(img_bytes)
        hasher.update(text_content.encode('utf-8'))
        return hasher.hexdigest()

    def _get_freq_features(self, image_path):
        """
        Extract frequency domain features (matching the training script's FFT logic).
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return torch.zeros(1024)
            img = cv2.resize(img, (128, 128))
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
            magnitude_spectrum = np.nan_to_num(magnitude_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
            feat_map = cv2.resize(magnitude_spectrum, (32, 32))
            std_val = feat_map.std()
            if std_val < 1e-6 or np.isnan(std_val): return torch.zeros(1024)
            feat_map = (feat_map - feat_map.mean()) / (std_val + 1e-8)
            feat_map = np.clip(feat_map, -10.0, 10.0)
            return torch.tensor(feat_map.flatten(), dtype=torch.float32)
        except Exception:
            return torch.zeros(1024)

    def _get_ocr_overlap(self, image_path, text):
        if self.ocr_reader is None:
            return torch.tensor([0.0, 0.0], dtype=torch.float32)
        try:
            result = self.ocr_reader.ocr(image_path)
            img_tokens = []
            if result and result[0]:
                for line in result[0]:
                    img_tokens.append(line[1][0].lower())
            
            caption_tokens = set(text.lower().split())
            if not img_tokens or not caption_tokens:
                return torch.tensor([0.0, 0.0], dtype=torch.float32)
            
            matches = 0
            for i_tok in img_tokens:
                if len(i_tok) < 2: continue
                for c_tok in caption_tokens:
                    if i_tok in c_tok or c_tok in i_tok:
                        matches += 1
                        break
            
            img_overlap = min(matches / len(img_tokens), 1.0)
            cap_overlap = min(matches / len(caption_tokens), 1.0)
            return torch.tensor([img_overlap, cap_overlap], dtype=torch.float32)
        except Exception:
            return torch.tensor([0.0, 0.0], dtype=torch.float32)

    def run_level1(self, image_path, text_content):
        """Level 1 - Data Integrity Defense (Hash-based)"""
        current_hash = self._calculate_hash(image_path, text_content)
        if current_hash in self.hash_registry:
            return True, "L1 Success: Hash matched (Data is from trusted knowledge base)"
        return False, "L1 Failed: Hash mismatch or new data"

    def run_level2(self, image_path, text_content):
        """Level 2 - Perceptual Consistency Defense (Signal-level Check)"""
        try:
            with self.gpu_lock:
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                text_tensor = clip.tokenize(text_content, truncate=True).to(self.device)
                ocr_score = self._get_ocr_overlap(image_path, text_content).unsqueeze(0).to(self.device)
                freq_feat = self._get_freq_features(image_path).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.l2_model(image_tensor, text_tensor, ocr_score, freq_feat)
                    prob = torch.sigmoid(output).item()
                
            if prob > 0.5:
                return True, f"L1/L2 Success: Signal consistency verified (Score: {prob:.4f})"
            else:
                return False, f"L2 Failed: Abnormal signal or noise detected (Score: {prob:.4f})"
        except Exception as e:
            # Fallback to Level 3 if Level 2 fails (e.g. model issues)
            return True, f"L2 Warning: Level 2 check failed due to error ({str(e)}), falling back to Level 3"

    def run_level3(self, image_path, text_content, query="Is the image content consistent with the text description?"):
        """Level 3 - Semantic Logic Defense (VLM-based Logic Auditor)"""
        prompt = (
            f"You are a strict consistency auditor for a Multi-modal RAG system.\n"
            f"Retrieved Text Context: {text_content}\n"
            f"Task: Audit if the visual information in the provided image is logically consistent with the text description.\n"
            f"Look for semantic contradictions (e.g., text describes a rising trend but image shows a falling one).\n"
            f"Respond with a JSON object: {{'is_safe': boolean, 'reason': string, 'final_response': string}}.\n"
            f"If safe, final_response should be 'Information verified'. If unsafe, final_response should explain the contradiction."
        )
        
        message = ChatMessage(
            role="user",
            blocks=[
                TextBlock(text=prompt),
                ImageBlock(path=image_path)
            ]
        )
        
        try:
            response = self.vlm.chat([message])
            content = response.message.content.strip()
            # Clean JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            return result.get('is_safe', False), result.get('reason', 'No reason'), result.get('final_response', '')
        except Exception as e:
            return False, f"L3 Error: {str(e)}", "Audit failed"

    def process(self, image_path, text_content, query=None):
        """
        Main entry point for the defensive pipeline.
        """
        # Level 1
        l1_safe, l1_msg = self.run_level1(image_path, text_content)
        if l1_safe:
            return True, "Level 1", "Information verified (Trusted source)"

        # Level 2
        l2_safe, l2_msg = self.run_level2(image_path, text_content)
        if not l2_safe:
            return False, "Level 2", f"Refused: {l2_msg}"

        # Level 3
        l3_safe, l3_reason, l3_response = self.run_level3(image_path, text_content, query)
        if not l3_safe:
            return False, "Level 3", f"Refused: {l3_reason}"
        
        return True, "Level 3", l3_response

if __name__ == "__main__":
    # Try to find project root by looking for 'dataset' directory
    if os.path.exists("dataset"):
        base_dir = "."
    elif os.path.exists("../dataset"):
        base_dir = ".."
    else:
        base_dir = "../.."

    # Quick test
    pipeline = DefensiveRAGPipeline()
    
    # Test with a known clean pair
    test_image = os.path.join(base_dir, "dataset/images_merged/img_000001.jpg")
    test_text = "A scatter plot with a grid background displays seven labeled data points, each marked by a colored dot and connected by gray arrows indicating directional relationships. The green dots represent “Total response,” “Frequent interaction,” and “Total interaction,” while the blue dots indicate “Regularly interaction,” “Major response,” “2 codes,” and “No interaction.” Arrows originate from “Total response” to “Frequent interaction” and “Total interaction”; from “Frequent interaction” to “Total interaction”; and from “Total interaction” to both “Regularly interaction” and “Major response.” A separate arrow connects “Regularly interaction” back to “Major response.” The point labeled “2 codes” is isolated with no incoming or outgoing connections, as is “No interaction,” which lies below and to the right of all other points. Numerical annotations “1,” “2,” and “3” appear near the arrows between “Total response,” “Frequent interaction,” and “Total interaction,” suggesting sequence or weighting. The spatial layout implies a progression from lower-left (“Total response”) toward upper-right clusters, with “No interaction” positioned farthest in the bottom-right quadrant."
    
    is_safe, risk_level, final_response = pipeline.process(test_image, test_text)
    print(f"Test Result:\nSafe: {is_safe}\nLevel: {risk_level}\nResponse: {final_response}")
