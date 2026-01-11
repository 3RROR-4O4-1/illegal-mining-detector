"""
Dataset building: statistics extraction, distribution matching, and augmentation.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import json
import shutil
from typing import Dict, List, Optional, Tuple
import random


def extract_statistics(image_dir: str, sample_pixels: int = 100000) -> Dict:
    """
    Extract RGB statistics from a directory of images.
    Uses reservoir sampling for memory efficiency.
    
    Args:
        image_dir: Directory containing JPG/PNG images
        sample_pixels: Max pixels to sample (default 100k uses ~1.2MB)
        
    Returns:
        Dictionary with per-channel statistics
    """
    image_path = Path(image_dir)
    image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))
    
    if not image_files:
        raise ValueError(f"No images found in {image_dir}")
    
    # Pre-allocate fixed-size reservoir (memory-efficient)
    reservoir = np.zeros((sample_pixels, 3), dtype=np.float32)
    total_seen = 0
    
    for img_file in image_files:
        img = np.array(Image.open(img_file).convert("RGB"), dtype=np.float32) / 255.0
        pixels = img.reshape(-1, 3)  # Flatten to (N, 3)
        
        for px in pixels:
            if total_seen < sample_pixels:
                reservoir[total_seen] = px
            else:
                # Reservoir sampling: replace with probability sample_pixels/total_seen
                j = random.randint(0, total_seen)
                if j < sample_pixels:
                    reservoir[j] = px
            total_seen += 1
        
        del img, pixels  # Free memory
    
    # Use only filled portion
    n = min(total_seen, sample_pixels)
    samples = reservoir[:n]
    
    stats = {
        "r_mean": float(np.mean(samples[:, 0])),
        "r_std": float(np.std(samples[:, 0])),
        "r_p2": float(np.percentile(samples[:, 0], 2)),
        "r_p98": float(np.percentile(samples[:, 0], 98)),
        
        "g_mean": float(np.mean(samples[:, 1])),
        "g_std": float(np.std(samples[:, 1])),
        "g_p2": float(np.percentile(samples[:, 1], 2)),
        "g_p98": float(np.percentile(samples[:, 1], 98)),
        
        "b_mean": float(np.mean(samples[:, 2])),
        "b_std": float(np.std(samples[:, 2])),
        "b_p2": float(np.percentile(samples[:, 2], 2)),
        "b_p98": float(np.percentile(samples[:, 2], 98)),
        
        "n_images": len(image_files),
        "n_pixels_sampled": n,
        "source_dir": str(image_dir)
    }
    
    return stats


def match_image_to_stats(image: np.ndarray, target_stats: Dict) -> np.ndarray:
    """
    Transform an image to match target statistics (Landsat style).
    
    Args:
        image: RGB image as numpy array (uint8, 0-255)
        target_stats: Statistics from extract_statistics()
        
    Returns:
        Matched image as numpy array (uint8, 0-255)
    """
    img = image.astype(np.float32) / 255.0
    result = np.zeros_like(img)
    
    for i, channel in enumerate(["r", "g", "b"]):
        src = img[:, :, i]
        
        # Get source percentiles
        src_p2, src_p98 = np.percentile(src, [2, 98])
        
        # Get target percentiles
        tgt_p2 = target_stats[f"{channel}_p2"]
        tgt_p98 = target_stats[f"{channel}_p98"]
        
        # Map source to target distribution
        if src_p98 > src_p2:
            normalized = (src - src_p2) / (src_p98 - src_p2)
            result[:, :, i] = normalized * (tgt_p98 - tgt_p2) + tgt_p2
        else:
            result[:, :, i] = target_stats[f"{channel}_mean"]
    
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


class SatelliteAugmentation:
    """
    Domain-appropriate augmentation for satellite imagery.
    
    Unlike MNIST-style augmentation, satellite images need:
    - Only 90° rotations (not arbitrary angles)
    - Atmospheric haze simulation
    - Sensor noise
    - No elastic deformation
    """
    
    def __init__(self, strength: str = "medium"):
        """
        Args:
            strength: "light", "medium", or "strong"
        """
        self.strength = strength
        
        # Parameters per strength level
        self.params = {
            "light": {
                "brightness_range": (0.95, 1.05),
                "contrast_range": (0.95, 1.05),
                "noise_std": 0.01,
                "haze_prob": 0.1,
                "haze_strength": 0.05
            },
            "medium": {
                "brightness_range": (0.85, 1.15),
                "contrast_range": (0.85, 1.15),
                "noise_std": 0.02,
                "haze_prob": 0.3,
                "haze_strength": 0.1
            },
            "strong": {
                "brightness_range": (0.7, 1.3),
                "contrast_range": (0.7, 1.3),
                "noise_std": 0.03,
                "haze_prob": 0.5,
                "haze_strength": 0.15
            }
        }[strength]
    
    def augment(self, image: np.ndarray) -> np.ndarray:
        """Apply random augmentation to an image."""
        img = image.astype(np.float32) / 255.0
        
        # 1. Random 90° rotation
        k = random.randint(0, 3)
        img = np.rot90(img, k)
        
        # 2. Random flip
        if random.random() < 0.5:
            img = np.fliplr(img)
        if random.random() < 0.5:
            img = np.flipud(img)
        
        # 3. Brightness adjustment (sun angle simulation)
        brightness = random.uniform(*self.params["brightness_range"])
        img = img * brightness
        
        # 4. Contrast adjustment
        contrast = random.uniform(*self.params["contrast_range"])
        mean = img.mean()
        img = (img - mean) * contrast + mean
        
        # 5. Atmospheric haze
        if random.random() < self.params["haze_prob"]:
            haze = self.params["haze_strength"] * random.random()
            # Haze is more visible in darker areas
            luminance = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
            haze_mask = 1 - luminance
            for i in range(3):
                img[:, :, i] = img[:, :, i] + haze * haze_mask
        
        # 6. Sensor noise
        noise = np.random.normal(0, self.params["noise_std"], img.shape)
        img = img + noise
        
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
    
    def augment_batch(self, image: np.ndarray, n: int) -> List[np.ndarray]:
        """Generate n augmented versions of an image."""
        return [self.augment(image) for _ in range(n)]


def build_dataset(
    positive_dir: str,
    negative_dir: str,
    output_dir: str,
    target_stats: Optional[Dict] = None,
    augmentation_strength: str = "medium",
    n_augmented_per_image: int = 5,
    match_distributions: bool = True,
    val_split: float = 0.2
) -> Dict:
    """
    Build a complete training dataset with a PHYSICAL Train/Validation split.
    
    Structure:
        output_dir/
            train/
                positive/ (Originals + Augmented)
                negative/ (Originals + Augmented)
            val/
                positive/ (Originals only)
                negative/ (Originals only)
    """
    output_path = Path(output_dir)
    
    # 1. Prepare Directory Structure
    for split in ["train", "val"]:
        for label in ["positive", "negative"]:
            (output_path / split / label).mkdir(parents=True, exist_ok=True)
            
    # 2. Extract stats if needed
    if target_stats is None and match_distributions:
        print("  Extracting statistics from positive images...")
        target_stats = extract_statistics(positive_dir)
    
    augmenter = SatelliteAugmentation(strength=augmentation_strength)
    
    metadata = {
        "train_positive": 0, "train_negative": 0,
        "val_positive": 0, "val_negative": 0,
        "target_stats": target_stats
    }
    
    # 3. Processing Logic
    def process_class(source_dir, class_name, is_positive):
        files = list(Path(source_dir).glob("*.jpg")) + list(Path(source_dir).glob("*.png"))
        
        # SHUFFLE AND SPLIT
        random.shuffle(files)
        split_idx = int(len(files) * (1 - val_split))
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        print(f"  Processing {class_name}: {len(train_files)} training, {len(val_files)} validation")
        
        # --- TRAIN SET (Augmented) ---
        for i, img_file in enumerate(train_files):
            img = np.array(Image.open(img_file).convert("RGB"))
            
            # Distribution matching (for Forest images)
            if not is_positive and match_distributions and target_stats:
                img = match_image_to_stats(img, target_stats)
            
            # Save Original
            base_name = f"{class_name}_{i:05d}"
            Image.fromarray(img).save(output_path / "train" / class_name / f"{base_name}.jpg", quality=95)
            metadata[f"train_{class_name}"] += 1
            
            # Save Augmented versions
            for j in range(n_augmented_per_image):
                aug_img = augmenter.augment(img)
                aug_name = f"{base_name}_aug{j:02d}.jpg"
                Image.fromarray(aug_img).save(output_path / "train" / class_name / aug_name, quality=85)
                metadata[f"train_{class_name}"] += 1
                del aug_img
            del img

        # --- VAL SET ---
        for i, img_file in enumerate(val_files):
            img = np.array(Image.open(img_file).convert("RGB"))
            
            # Still apply distribution matching to validation negatives so they resemble the domain
            if not is_positive and match_distributions and target_stats:
                img = match_image_to_stats(img, target_stats)
                
            # Save Original Only
            val_name = f"{class_name}_{i:05d}.jpg"
            Image.fromarray(img).save(output_path / "val" / class_name / val_name, quality=95)
            metadata[f"val_{class_name}"] += 1
            del img

    # 4. Execute
    print("  Processing Positive (Mine) images...")
    process_class(positive_dir, "positive", is_positive=True)
    
    print("  Processing Negative (Forest) images...")
    process_class(negative_dir, "negative", is_positive=False)
    
    # 5. Save Metadata
    json_metadata = {k: v for k, v in metadata.items() if k != "target_stats"}
    if target_stats:
        json_metadata["target_stats"] = target_stats
    
    with open(output_path / "dataset_metadata.json", "w") as f:
        json.dump(json_metadata, f, indent=2)
    
    return metadata



def merge_image_directories(dirs: List[str], output_dir: str) -> int:
    """
    Merge multiple image directories into one.
    
    Args:
        dirs: List of source directories
        output_dir: Output directory
        
    Returns:
        Number of images copied
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for src_dir in dirs:
        src_path = Path(src_dir)
        if not src_path.exists():
            print(f"  Warning: {src_dir} does not exist, skipping")
            continue
        
        for img_file in list(src_path.glob("*.jpg")) + list(src_path.glob("*.png")):
            dst_name = f"img_{count:04d}{img_file.suffix}"
            shutil.copy(img_file, output_path / dst_name)
            count += 1
    
    return count
