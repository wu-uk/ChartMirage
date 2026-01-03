import os
import cv2
import numpy as np
import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Try to find project root by looking for 'dataset' directory
if os.path.exists("dataset"):
    base_dir = "."
elif os.path.exists("../../dataset"):
    base_dir = "../.."
else:
    base_dir = ".."

SOURCE_DIR = os.path.join(base_dir, "dataset/images_merged_fake")
TARGET_BASE_DIR = os.path.join(base_dir, "dataset/images_noise_fake")

# Augmentation types
AUG_TYPES = ["gaussian_noise", "salt_pepper_noise", "gaussian_blur", "rotation"]

def setup_directories():
    if not os.path.exists(TARGET_BASE_DIR):
        os.makedirs(TARGET_BASE_DIR)
    for aug in AUG_TYPES:
        path = os.path.join(TARGET_BASE_DIR, aug)
        if not os.path.exists(path):
            os.makedirs(path)

def add_gaussian_noise(image, sigma=25):
    row, col, ch = image.shape
    mean = 0
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, prob=0.02):
    output = np.copy(image)
    # Total pixels * channels
    # prob is total percentage of noise. split between salt and pepper.
    num_salt = np.ceil(prob * image.size * 0.5)
    # Generate coordinates for each dimension
    coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
    output[tuple(coords)] = 255
    
    num_pepper = np.ceil(prob * image.size * 0.5)
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
    output[tuple(coords)] = 0
    return output

def add_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def add_rotation(image, angle=15):
    row, col = image.shape[:2]
    center = (col / 2, row / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat, (col, row), borderMode=cv2.BORDER_REFLECT)

def process_single_image(img_path):
    try:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            return f"Failed to read {filename}"

        # 1. Gaussian Noise
        noisy_gauss = add_gaussian_noise(img)
        cv2.imwrite(os.path.join(TARGET_BASE_DIR, "gaussian_noise", filename), noisy_gauss)

        # 2. Salt and Pepper
        noisy_sp = add_salt_and_pepper_noise(img)
        cv2.imwrite(os.path.join(TARGET_BASE_DIR, "salt_pepper_noise", filename), noisy_sp)

        # 3. Blur
        blurred = add_gaussian_blur(img)
        cv2.imwrite(os.path.join(TARGET_BASE_DIR, "gaussian_blur", filename), blurred)

        # 4. Rotation
        rotated = add_rotation(img)
        cv2.imwrite(os.path.join(TARGET_BASE_DIR, "rotation", filename), rotated)
        
        return None
    except Exception as e:
        return f"Error processing {img_path}: {e}"

def main():
    setup_directories()
    
    # Find images
    extensions = ["*.png", "*.jpg", "*.jpeg"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))
    
    print(f"Found {len(image_files)} images in {SOURCE_DIR}")
    
    if not image_files:
        print("No images found. Exiting.")
        return

    # Process in parallel
    max_workers = min(os.cpu_count(), 16)
    print(f"Starting augmentation with {max_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_single_image, image_files), total=len(image_files)))
        
    errors = [r for r in results if r is not None]
    if errors:
        print(f"Completed with {len(errors)} errors.")
        for e in errors[:5]:
            print(e)
    else:
        print("All images processed successfully.")

if __name__ == "__main__":
    main()
