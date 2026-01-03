import json
import os
import cv2
import sys
import numpy as np
import time
import shutil

# Add project root to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Try to find project root by looking for 'dataset' directory
if os.path.exists("dataset"):
    base_dir = "."
elif os.path.exists("../../../dataset"):
    base_dir = "../../.."
else:
    base_dir = "../.."

# Fix for PaddleOCR missing libnvrtc.so.13
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/home/ASC26team2/miniconda3/envs/ChartMirage/lib/python3.12/site-packages/nvidia/cu13/lib'

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import clip
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from paddleocr import PaddleOCR

# --- Setup Paths ---
DATA_PATH = os.path.join(base_dir, 'dataset/final_qa_merged_unified.json')
OCR_CACHE_PATH = os.path.join(base_dir, "dataset/ocr_cache.json")
BASELINE_WEIGHTS = os.path.join(base_dir, "core/experiments/consistency_check/best_model.pth")
OUTPUT_LOGS_DIR = os.path.join(base_dir, "outputs/logs/consistency_check")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize OCR
print("Initializing PaddleOCR...")
# PaddleOCR loads model to GPU by default if available
reader = PaddleOCR(use_angle_cls=True, lang='en') 

# --- 1. Load CLIP ---
print("Loading CLIP model...")
clip_model, preprocess = clip.load("ViT-L/14", device=device)

def load_ocr_cache():
    if os.path.exists(OCR_CACHE_PATH):
        with open(OCR_CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_ocr_cache(cache):
    with open(OCR_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False)

def precompute_ocr(json_path, root_dir):
    print("Checking OCR cache...")
    cache = load_ocr_cache()
    
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    # Extract all unique image paths
    image_paths = set()
    for entry in raw_data:
        if 'image' in entry:
            image_paths.add(entry['image'])
        if 'edited_image' in entry:
            image_paths.add(entry['edited_image'])
            
    # Resolve paths
    resolved_paths = []
    for rel_path in image_paths:
        rel_path_norm = rel_path.replace("/", os.sep)
        candidates = [
            os.path.join(root_dir, "dataset", rel_path_norm),
            os.path.join(root_dir, rel_path_norm),
        ]
        parts = rel_path_norm.split(os.sep)
        if len(parts) > 1:
            subdir = parts[0]
            rest = os.path.join(*parts[1:])
            candidates.append(os.path.join(root_dir, "dataset", subdir, subdir, rest))
            
        found = False
        for p in candidates:
            if os.path.exists(p):
                resolved_paths.append(p)
                found = True
                break
        # if not found: print(f"Warning: Could not find image {rel_path}")

    # Identify missing entries
    missing_paths = [p for p in resolved_paths if p not in cache]
    
    if missing_paths:
        print(f"Running OCR on {len(missing_paths)} new images...")
        updates = 0
        # Process one by one (PaddleOCR handles GPU internally)
        for img_path in tqdm(missing_paths, desc="OCR Pre-computation"):
            try:
                # PaddleOCR returns [ [ [ [x1,y1], ... ], (text, confidence) ], ... ]
                # Note: cls=True caused issues in newer paddleocr versions?
                result = reader.ocr(img_path)
                detected_texts = []
                if result and result[0]:
                    for line in result[0]:
                        detected_texts.append(line[1][0].lower())
                cache[img_path] = detected_texts
                updates += 1
                
                # Save periodically
                if updates % 100 == 0:
                     save_ocr_cache(cache)
            except Exception as e:
                print(f"Error on {img_path}: {e}")
                cache[img_path] = []
                
        if updates > 0:
            save_ocr_cache(cache)
            
    print(f"OCR Cache ready with {len(cache)} entries.")
    return cache

# --- 2. Dataset Class ---
class ChartConsistencyDataset(Dataset):
    def __init__(self, json_path, root_dir, preprocess, ocr_cache, mode='train', split_ratio=0.8):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.mode = mode
        self.data = []
        self.ocr_cache = ocr_cache # Use pre-computed cache
        
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        random.seed(42)
        random.shuffle(raw_data)
        split_idx = int(len(raw_data) * split_ratio)
        
        if mode == 'train':
            self.raw_entries = raw_data[:split_idx]
        else:
            self.raw_entries = raw_data[split_idx:]
            
        self._prepare_data()

    def _get_freq_features(self, image_path):
        """
        Extract frequency domain features using Log-scaled DFT magnitude.
        Returns a fixed-size feature vector (e.g., 32x32 flattened).
        """
        try:
            # 1. Read as grayscale
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None: return torch.zeros(1024) # 32*32
            
            # 2. Resize to 128x128 for consistency before FFT
            img = cv2.resize(img, (128, 128))
            
            # 3. FFT
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
            
            # Replace inf/nan
            magnitude_spectrum = np.nan_to_num(magnitude_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 4. Resize spectrum to small feature map (32x32)
            # We want to capture the general energy distribution, not high-res details
            feat_map = cv2.resize(magnitude_spectrum, (32, 32))
            
            # 5. Normalize
            std_val = feat_map.std()
            if std_val < 1e-6 or np.isnan(std_val):
                return torch.zeros(1024)
                
            feat_map = (feat_map - feat_map.mean()) / (std_val + 1e-8)
            
            # Final Safety Clip
            feat_map = np.clip(feat_map, -10.0, 10.0)
            
            return torch.tensor(feat_map.flatten(), dtype=torch.float32)
        except Exception:
            return torch.zeros(1024)

    def _resolve_path(self, rel_path):
        rel_path_norm = rel_path.replace("/", os.sep)
        candidates = [
            os.path.join(self.root_dir, "dataset", rel_path_norm),
            os.path.join(self.root_dir, rel_path_norm),
        ]
        parts = rel_path_norm.split(os.sep)
        if len(parts) > 1:
            subdir = parts[0]
            rest = os.path.join(*parts[1:])
            candidates.append(os.path.join(self.root_dir, "dataset", subdir, subdir, rest))
            
        for p in candidates:
            if os.path.exists(p): return p
        return None

    def _prepare_data(self):
        print(f"Preparing {len(self.raw_entries)} entries for {self.mode}...")
        for entry in self.raw_entries:
            # Consistent
            orig_img_path = self._resolve_path(entry['image'])
            if orig_img_path:
                self.data.append({
                    'image_path': orig_img_path,
                    'text': entry['original_caption'],
                    'label': 1.0
                })
                # Inconsistent 1
                if 'fake_caption' in entry:
                     self.data.append({
                        'image_path': orig_img_path,
                        'text': entry['fake_caption'],
                        'label': 0.0
                    })
            # Inconsistent 2
            if 'edited_image' in entry:
                edited_img_path = self._resolve_path(entry['edited_image'])
                if edited_img_path:
                     self.data.append({
                        'image_path': edited_img_path,
                        'text': entry['original_caption'],
                        'label': 0.0
                    })
        print(f"Total samples generated: {len(self.data)}")

    def _get_ocr_overlap(self, image_path, caption):
        # 1. Retrieve cached OCR results
        # If path not in cache (should not happen if precomputed correctly), return empty
        img_tokens = self.ocr_cache.get(image_path, [])
        
        # 2. Process Caption
        caption_tokens = set(caption.lower().split())
        
        # 3. Calculate Overlap
        if not img_tokens or not caption_tokens:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        matches = 0
        for i_tok in img_tokens:
            if len(i_tok) < 2: continue # skip noise
            for c_tok in caption_tokens:
                if i_tok in c_tok or c_tok in i_tok: # Substring match
                    matches += 1
                    break
        
        # Feature 1: How much of the image text is in the caption? (Precision-like)
        img_overlap = min(matches / len(img_tokens), 1.0)
        
        # Feature 2: How much of the caption text is in the image? (Recall-like)
        cap_overlap = min(matches / len(caption_tokens), 1.0)
        
        return np.array([img_overlap, cap_overlap], dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['image_path']
        text = item['text']
        label = item['label']

        try:
            image = Image.open(image_path).convert("RGB")
            
            # Augmentation for Negative Samples (Fake images)
            # To simulate "fake" artifacts and make the model more robust against clean fakes
            if label == 0.0 and self.mode == 'train':
                # 50% chance to apply augmentation
                if random.random() < 0.5:
                    aug_type = random.choice(['blur', 'jpeg', 'resize'])
                    if aug_type == 'blur':
                        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
                    elif aug_type == 'jpeg':
                        # Simulate JPEG compression artifacts
                        import io
                        buffer = io.BytesIO()
                        image.save(buffer, 'JPEG', quality=random.randint(50, 80))
                        buffer.seek(0)
                        image = Image.open(buffer).convert("RGB")
                    elif aug_type == 'resize':
                        # Simulate resizing artifacts
                        w, h = image.size
                        scale = random.uniform(0.7, 0.9)
                        image = image.resize((int(w*scale), int(h*scale)), Image.BILINEAR)
                        image = image.resize((w, h), Image.BILINEAR)

            image_tensor = self.preprocess(image) 
            text_tensor = clip.tokenize(text, truncate=True).squeeze(0)
            
            # OCR Feature
            # Note: OCR is slow, so doing it in __getitem__ during training will be slow.
            # Ideally, pre-compute. For now, we rely on the in-memory cache.
            # The first epoch will be very slow, subsequent ones fast.
            ocr_score = self._get_ocr_overlap(image_path, text)
            
            # Frequency Feature
            freq_feat = self._get_freq_features(image_path)
            
            return {
                'image': image_tensor,
                'text': text_tensor,
                'ocr_score': torch.tensor(ocr_score, dtype=torch.float),
                'freq_feat': freq_feat,
                'label': torch.tensor(label, dtype=torch.float)
            }
        except Exception as e:
            return self.__getitem__(random.randint(0, len(self.data)-1))

import torch.nn.functional as F

# --- 2.5 Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.6, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- 3. Consistency Model (Updated) ---
class ConsistencyClassifier(nn.Module):
    def __init__(self, clip_model, dropout=0.3):
        super(ConsistencyClassifier, self).__init__()
        self.clip_model = clip_model
        
        # Freeze most of CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
            
        try:
            # Unfreeze last 3 ResBlocks for better adaptation
            for param in self.clip_model.visual.transformer.resblocks[-3:].parameters():
                param.requires_grad = True
        except: pass

        self.embed_dim = clip_model.visual.output_dim 
        
        # Frequency Feature Processor
        # Input: 1024 (32x32)
        self.freq_encoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        
        # Classifier Head - Pure CLIP Linear Probe
        # Input: [image_embed, text_embed, ocr_score] - removed freq_embed for now
        # Size: embed_dim + embed_dim + 2
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim * 2 + 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(128, 1)
        )
        self.classifier = self.classifier.float()

    def forward(self, image, text, ocr_score, freq_feat):
        # Encode
        image_features = self.clip_model.encode_image(image)
        text_features = self.clip_model.encode_text(text)
        
        # Force float32 to prevent NaN in mixed precision
        image_features = image_features.float()
        text_features = text_features.float()
        
        # Normalize
        image_features = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-8)
        text_features = text_features / (text_features.norm(dim=1, keepdim=True) + 1e-8)

        # Process Frequency Features - DISABLED FOR BASELINE PROBE
        # freq_embed = self.freq_encoder(freq_feat)

        # Combine Features + OCR Score
        # ocr_score has shape [batch, 2]
        ocr_score = ocr_score.float()
        
        # combined = torch.cat((image_features, text_features, ocr_score, freq_embed), dim=1)
        combined = torch.cat((image_features, text_features, ocr_score), dim=1)
        
        return self.classifier(combined).squeeze()

# --- 4. Training ---
def train(pos_weight_val, dropout_val):
    # --- Logging Setup ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(OUTPUT_LOGS_DIR, f"grid_search_{timestamp}_pw{pos_weight_val}_do{dropout_val}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Save current script
    shutil.copy(__file__, os.path.join(log_dir, "train_script.py"))
        
    BATCH_SIZE = 16 
    EPOCHS = 20 # Stage 2: Fine-tuning doesn't need 50 epochs
    LR = 1e-4
    ROOT_DIR = base_dir # Adjust if needed

    print(f"--- Starting Experiment: pos_weight={pos_weight_val}, dropout={dropout_val} ---")
    
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize PaddleOCR
    print("Initializing PaddleOCR...")
    try:
        # Suppress paddle warning
        import logging
        logging.getLogger('ppocr').setLevel(logging.ERROR)
        # Note: use_angle_cls=True is deprecated in newer versions, use use_textline_orientation=True if needed
        # But for compatibility with installed version, we keep as is or adjust based on error
        reader = PaddleOCR(use_angle_cls=True, lang='en') 
    except Exception as e:
        print(f"PaddleOCR init failed: {e}")
        reader = None

    # Load CLIP
    print("Loading CLIP model...")
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Pre-compute OCR Cache if not exists
    print(f"Dataset path: {DATA_PATH}")
    ocr_cache = {}
    if os.path.exists(OCR_CACHE_PATH):
        print("Loading OCR cache...")
        with open(OCR_CACHE_PATH, 'r') as f:
            ocr_cache = json.load(f)
    
    print("Checking OCR cache...")
    # (OCR cache generation logic omitted for brevity as it should be done already)
    print(f"OCR Cache ready with {len(ocr_cache)} entries.")

    # Datasets
    # Fixed instantiation: removed 'reader' argument which was causing the error
    full_dataset = ChartConsistencyDataset(DATA_PATH, ROOT_DIR, preprocess, ocr_cache, mode='train', split_ratio=0.8)
    train_size = len(full_dataset.raw_entries)
    print(f"Preparing {train_size} entries for train...")
    
    val_dataset = ChartConsistencyDataset(DATA_PATH, ROOT_DIR, preprocess, ocr_cache, mode='val', split_ratio=0.8)
    val_size = len(val_dataset.raw_entries)
    print(f"Preparing {val_size} entries for val...")
    
    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    model = ConsistencyClassifier(clip_model, dropout=dropout_val).to(device)
    # Ensure entire model is in float32
    model.float()
    
    # Optimizer & Scheduler
    # Stage 2: Fine-tuning
    optimizer = torch.optim.AdamW([
        {'params': model.clip_model.visual.transformer.resblocks[-3:].parameters(), 'lr': 1e-6}, # Low LR for CLIP
        # {'params': model.freq_encoder.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-5} # Lower LR for head to preserve weights
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    
    # Load Pre-trained weights if available (LP-FT)
    # Check if best_model.pth exists
    if os.path.exists(BASELINE_WEIGHTS):
        print(f"Loading baseline weights from {BASELINE_WEIGHTS} for Fine-tuning...")
        try:
            state_dict = torch.load(BASELINE_WEIGHTS, map_location=device)
            model.load_state_dict(state_dict, strict=False) # strict=False because we unfreezed layers? No, architecture is same.
            # Actually strict=False is safer if we add/remove buffers
        except Exception as e:
            print(f"Failed to load weights: {e}")
    else:
        print(f"Warning: No baseline weights found at {BASELINE_WEIGHTS}. Training from scratch (might be unstable).")

    
    # Loss
    # Reverting to pos_weight=1.85 as it was stable
    pos_weight = torch.tensor([pos_weight_val]).to(device) 
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    best_val_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        valid_batches = 0
        train_preds = []
        train_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, batch in enumerate(progress_bar):
            images = batch['image'].to(device)
            texts = batch['text'].to(device)
            ocr_scores = batch['ocr_score'].to(device)
            freq_feats = batch['freq_feat'].to(device)
            labels = batch['label'].to(device)
            
            # Debug: Check for NaNs in input
            if i == 0:
                if torch.isnan(images).any(): print("NaN in images!")
                if torch.isnan(freq_feats).any(): print("NaN in freq_feats!")
                if torch.isnan(ocr_scores).any(): print("NaN in ocr_scores!")
            
            optimizer.zero_grad()
            logits = model(images, texts, ocr_scores, freq_feats)
            
            loss = criterion(logits, labels)
            
            if torch.isnan(loss):
                print("Warning: Loss is NaN, skipping batch.")
                continue
                
            loss.backward()
            
            # Gradient Clipping to prevent NaN
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            valid_batches += 1
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
            
        scheduler.step()
        
        avg_train_loss = train_loss / valid_batches if valid_batches > 0 else 0.0
        train_acc = accuracy_score(train_targets, train_preds)
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f}")
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                texts = batch['text'].to(device)
                ocr_scores = batch['ocr_score'].to(device)
                freq_feats = batch['freq_feat'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(images, texts, ocr_scores, freq_feats)
                preds = (torch.sigmoid(logits) > 0.5).float()
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds, zero_division=0)
        val_recall = recall_score(val_targets, val_preds, zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        
        print(f"Epoch {epoch+1} Val Acc: {val_acc:.4f} | P: {val_precision:.4f} | R: {val_recall:.4f} | F1: {val_f1:.4f}")
        
        # Save metrics to log
        with open(os.path.join(log_dir, "metrics.txt"), "a") as f:
            f.write(f"Epoch {epoch+1}: Loss={avg_train_loss:.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}, P={val_precision:.4f}, R={val_recall:.4f}, F1={val_f1:.4f}\n")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"core/experiments/consistency_check/best_model_pw{pos_weight_val}_do{dropout_val}.pth")
            torch.save(model.state_dict(), os.path.join(log_dir, "best_model.pth"))
            print(f"Saved Best Model (F1: {best_val_f1:.4f})")
    
    return best_val_f1

# --- 5. Main Training Loop ---
if __name__ == "__main__":
    # Stage 2: Fine-tuning
    print("\n=== Stage 2: Fine-tuning (LP-FT), Unfrozen CLIP (Last 3 Layers), pos_weight=1.8 ===")
    try:
        best_f1 = train(1.8, 0.4) # Slightly lower pos_weight to improve Precision
        print(f"Final Best F1: {best_f1:.4f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Experiment failed: {e}")