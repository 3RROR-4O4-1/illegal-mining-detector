"""
Configuration parameters for the illegal mining detection pipeline.
Edit these values to customize the pipeline behavior.
"""

# =============================================================================
# DATA PATHS
# =============================================================================

# Local mine images (JPG files from Landsat)
LOCAL_MINE_DIR = "GreenAI/src/data/landsat_converted/2019/barragem_jpg"

# CSV with mine coordinates (lon, lat columns)
MINE_COORDINATES_CSV = "GreenAI/src/data/zones_centroids.csv"

# Output directory for all pipeline outputs
OUTPUT_DIR = "GreenAI/src/output"

# =============================================================================
# DATA COLLECTION SETTINGS
# =============================================================================

# Filter to only use coordinates within Brazil
FILTER_BRAZIL_ONLY = True

# Brazil bounding box (lon_min, lon_max, lat_min, lat_max)
BRAZIL_BOUNDS = (-75, -35, -35, 5)

# Number of forest (negative) samples to generate
N_FOREST_SAMPLES = 10  # Reduced from 500 for lower RAM usage

# Sample from protected areas for cleaner negatives
USE_PROTECTED_AREAS = True

# Satellite imagery settings (Landsat)
DATE_RANGE = "2023-01-01/2025-10-31"
MAX_CLOUD_COVER = 20

# Image size - distance from center point in km
# 2.5 km gives roughly 5km x 5km patches (similar to your reference image)
# At Landsat resolution (~30m/pixel), this is roughly 160-170 pixels
IMAGE_SIZE_KM = 2.5

# =============================================================================
# DATASET BUILDING SETTINGS
# =============================================================================

# Match fetched images to local Landsat style
MATCH_TO_LANDSAT_STYLE = True

# Augmentation settings
AUGMENTATION_STRENGTH = "medium"  # "light", "medium", "strong"
N_AUGMENTED_PER_IMAGE = 5  # Reduced from 5 for lower RAM usage

# =============================================================================
# MODEL TRAINING SETTINGS
# =============================================================================

# Model architecture
BACKBONE = "resnet34"  # "resnet18", "resnet34", "efficientnet_b0"

# Training hyperparameters
BATCH_SIZE = 16  # Reduced from 32 for lower RAM usage
LEARNING_RATE = 1e-4
EPOCHS = 300
EARLY_STOPPING_PATIENCE = 10
VALIDATION_SPLIT = 0.15

# Image size for model input
MODEL_IMAGE_SIZE = 224

# =============================================================================
# INFERENCE SETTINGS
# =============================================================================

# Confidence threshold for mining detection
MINING_THRESHOLD = 0.5

# =============================================================================
# SEGMENTATION DETECTION SETTINGS (Stage 1)
# =============================================================================

# SegFormer model (from HuggingFace)
SEGFORMER_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"

# Minimum area in pixels for a candidate zone
MIN_CANDIDATE_AREA = 500

# Confidence threshold for segmentation
SEGMENTATION_THRESHOLD = 0.3

# =============================================================================
# VALIDATION SETTINGS
# =============================================================================

# Path to CSV with known mining coordinates for validation
# Expected columns: lat, lon, label (where label is "mining" or "forest")
VALIDATION_CSV = "GreenAI/src/data/known_mining_sites.csv"
