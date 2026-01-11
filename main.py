import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import shutil
import time
import numpy as np
import gc

# Import configuration
from config import (
    LOCAL_MINE_DIR,
    MINE_COORDINATES_CSV,
    OUTPUT_DIR,
    BRAZIL_BOUNDS,
    N_FOREST_SAMPLES,
    DATE_RANGE,
    MAX_CLOUD_COVER,
    IMAGE_SIZE_KM,
    MATCH_TO_LANDSAT_STYLE,
    AUGMENTATION_STRENGTH,
    N_AUGMENTED_PER_IMAGE,
    BACKBONE,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    VALIDATION_SPLIT,
    MODEL_IMAGE_SIZE,
    MINING_THRESHOLD,
    VALIDATION_CSV,
)

from dataset_builder import extract_statistics, build_dataset
from classifier import train_model, Predictor

# Default radius for overview (10km from center = 20km x 20km area)
OVERVIEW_RADIUS_KM = 10.0


# =============================================================================
# STEP 1: Collect Mine Images (Positive Samples)
# =============================================================================

def step1_collect_mines() -> Dict:
    print("\n" + "=" * 60)
    print("STEP 1: Collecting Mine Images (Positive Samples)")
    print("=" * 60)
    
    output_path = Path(OUTPUT_DIR) / "raw_data" / "mines"
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    local_path = Path(LOCAL_MINE_DIR).resolve()
    csv_path = Path(MINE_COORDINATES_CSV).resolve()
    
    print(f"  Local mine directory: {local_path}")
    
    local_images = []
    local_image_size = None
    
    if local_path.exists():
        print("  Scanning for *_512.jpg files...")
        local_images = list(local_path.glob("*_512.jpg"))
        
        if not local_images:
            print("  ⚠ No *_512.jpg files found. Falling back to all *.jpg files.")
            local_images = list(local_path.glob("*.jpg")) + list(local_path.glob("*.png"))
        else:
            print(f"  ✓ Found {len(local_images)} specific 512px images.")

        if local_images:
            from PIL import Image
            first_img = Image.open(local_images[0])
            local_image_size = first_img.size
            print(f"  Local image size: {local_image_size[0]}x{local_image_size[1]} pixels")
    
    if local_images:
        print(f"  Copying {len(local_images)} local images to output...")
        for i, img_file in enumerate(local_images):
            dst_name = f"mine_local_{i:04d}.jpg"
            shutil.copy(img_file, output_path / dst_name)
            if (i + 1) % 100 == 0:
                print(f"    Copied {i + 1}/{len(local_images)}")
    else:
        print("  ⚠ No local images found!")

    landsat_stats = None
    if local_images:
        print("  Extracting color statistics from selected images...")
        landsat_stats = extract_statistics(str(output_path))
        
        stats_file = Path(OUTPUT_DIR) / "landsat_stats.json"
        with open(stats_file, "w") as f:
            json.dump(landsat_stats, f, indent=2)
        print(f"  ✓ Saved statistics to {stats_file}")

    fetched_count = 0
    coordinates = []
    
    if csv_path.exists():
        print(f"\n  Loading coordinates from CSV: {csv_path}")
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            lat_col = next((col for col in reader.fieldnames if col.lower() in ['lat', 'latitude', 'y']), None)
            lon_col = next((col for col in reader.fieldnames if col.lower() in ['lon', 'lng', 'longitude', 'x']), None)
            
            if lat_col and lon_col:
                f.seek(0)
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        lat, lon = float(row[lat_col]), float(row[lon_col])
                        b = BRAZIL_BOUNDS
                        if not (b[0] < lon < b[1] and b[2] < lat < b[3]):
                            continue
                        coordinates.append({"lat": lat, "lon": lon})
                    except ValueError:
                        continue
    
    if coordinates:
        print(f"  Fetching {len(coordinates)} additional samples from CSV coordinates...")
        
        try:
            from satellite_fetcher import SatelliteFetcher
            from PIL import Image
            
            fetcher = SatelliteFetcher(date_range=DATE_RANGE, max_cloud_cover=MAX_CLOUD_COVER)
            
            for i, coord in enumerate(coordinates):
                rgb, metadata = fetcher.fetch_image(
                    lat=coord["lat"], lon=coord["lon"],
                    distance_km=IMAGE_SIZE_KM, target_stats=landsat_stats
                )
                
                if rgb is not None:
                    filename = f"mine_fetched_{fetched_count:04d}.jpg"
                    Image.fromarray(rgb).save(output_path / filename, quality=95)
                    fetched_count += 1
                    
                if (i + 1) % 10 == 0:
                    print(f"    Processed {i + 1}/{len(coordinates)} ({fetched_count} successful)")
                
                time.sleep(0.5)
                
        except ImportError as e:
            print(f"  ⚠ Could not fetch satellite images: {e}")

    result = {
        "output_dir": str(output_path),
        "local_count": len(local_images),
        "fetched_count": fetched_count,
        "total_count": len(local_images) + fetched_count,
        "landsat_stats": landsat_stats,
        "image_size": local_image_size
    }
    
    print(f"\n  ✓ Total mine images collected: {result['total_count']}")
    return result


# =============================================================================
# STEP 2: Collect Forest Images (Negative Samples)
# =============================================================================

def step2_collect_forest(landsat_stats: Dict) -> Dict:
    print("\n" + "=" * 60)
    print("STEP 2: Collecting Forest Images (Negative Samples)")
    print("=" * 60)
    
    output_path = Path(OUTPUT_DIR) / "raw_data" / "forest"
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        from satellite_fetcher import SatelliteFetcher
        
        fetcher = SatelliteFetcher(date_range=DATE_RANGE, max_cloud_cover=MAX_CLOUD_COVER)
        
        print(f"  Generating {N_FOREST_SAMPLES} forest samples...")
        
        samples = fetcher.generate_forest_samples(
            n_samples=N_FOREST_SAMPLES,
            output_dir=str(output_path),
            target_stats=landsat_stats,
            brazil_bounds=BRAZIL_BOUNDS,
            distance_km=IMAGE_SIZE_KM
        )
        
        result = {"output_dir": str(output_path), "count": len(samples)}
        print(f"\n  ✓ Generated {len(samples)} forest images")
        
    except ImportError as e:
        print(f"  Warning: Could not fetch forest images: {e}")
        result = {"output_dir": str(output_path), "count": 0}
    
    return result


# =============================================================================
# STEP 3: Build Training Dataset
# =============================================================================

def step3_build_dataset(mines_dir: str, forest_dir: str, landsat_stats: Dict) -> Dict:
    print("\n" + "=" * 60)
    print("STEP 3: Building Training Dataset (Split First -> Augment Train Only)")
    print("=" * 60)
    
    output_path = Path(OUTPUT_DIR) / "dataset"
    
    metadata = build_dataset(
        positive_dir=mines_dir,
        negative_dir=forest_dir,
        output_dir=str(output_path),
        target_stats=landsat_stats,
        augmentation_strength=AUGMENTATION_STRENGTH,
        n_augmented_per_image=N_AUGMENTED_PER_IMAGE,
        match_distributions=MATCH_TO_LANDSAT_STYLE,
        val_split=VALIDATION_SPLIT 
    )
    
    metadata["output_dir"] = str(output_path)
    print(f"\n  ✓ Dataset built at {output_path}")
    
    return metadata

# =============================================================================
# STEP 4: Train Model
# =============================================================================

def step4_train(dataset_dir: str) -> Dict:
    print("\n" + "=" * 60)
    print("STEP 4: Training Classifier")
    print("=" * 60)
    
    output_path = Path(OUTPUT_DIR) / "model"
    
    history = train_model(
        dataset_dir=dataset_dir,
        output_dir=str(output_path),
        backbone=BACKBONE,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
        image_size=MODEL_IMAGE_SIZE
    )
    
    history["model_dir"] = str(output_path)
    print(f"\n  ✓ Model saved to {output_path}")
    
    return history



# =============================================================================
# STEP 5: Two-Stage Detection (Center + Radius)
# =============================================================================

def pixel_to_geo(pixel_coords: tuple, image_size: tuple, image_bounds: tuple) -> tuple:
    """Convert pixel coordinates to geographic coordinates."""
    x, y = pixel_coords
    w, h = image_size
    lon_min, lat_min, lon_max, lat_max = image_bounds
    
    lon = lon_min + (x / w) * (lon_max - lon_min)
    lat = lat_max - (y / h) * (lat_max - lat_min)
    
    return (lon, lat)


def bbox_to_geo(bbox: tuple, image_size: tuple, image_bounds: tuple) -> dict:
    """Convert pixel bounding box to geographic coordinates."""
    x1, y1, x2, y2 = bbox
    w, h = image_size
    lon_min, lat_min, lon_max, lat_max = image_bounds
    
    geo_lon_min = lon_min + (x1 / w) * (lon_max - lon_min)
    geo_lon_max = lon_min + (x2 / w) * (lon_max - lon_min)
    geo_lat_max = lat_max - (y1 / h) * (lat_max - lat_min)
    geo_lat_min = lat_max - (y2 / h) * (lat_max - lat_min)
    
    return {
        "lon_min": geo_lon_min, "lon_max": geo_lon_max,
        "lat_min": geo_lat_min, "lat_max": geo_lat_max,
        "centroid_lon": (geo_lon_min + geo_lon_max) / 2,
        "centroid_lat": (geo_lat_min + geo_lat_max) / 2
    }


def step5_detect_overview(
    model_dir: str,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    radius_km: float = OVERVIEW_RADIUS_KM,
    overview_max_dimension: int = 2048,
    classification_distance_km: float = 2.5,
    landsat_stats: Optional[Dict] = None,
    rate_limit_seconds: float = 0.5
) -> Dict:
    print("\n" + "=" * 60)
    print("STEP 5: Grid-Based Detection (Restored Classes)")
    print("=" * 60)

    model_path = Path(model_dir) / "best_model.pth"
    if not model_path.exists(): return {"error": "Classifier model not found"}
    
    try:
        from segmentation_detector import MiningSegmentationDetector, DetectionVisualizer
        from satellite_fetcher import SatelliteFetcher, center_to_bbox
        from classifier import Predictor
        from PIL import Image, ImageDraw
        import torch
    except ImportError as e: return {"error": str(e)}

    # 1. Fetch Overview
    print("\n  [Stage 1] Fetching Overview...")
    fetcher = SatelliteFetcher(date_range=DATE_RANGE, max_cloud_cover=MAX_CLOUD_COVER)
    overview_image, overview_metadata = fetcher.fetch_overview(
        center_lat=center_lat, center_lon=center_lon,
        radius_km=radius_km, max_dimension=overview_max_dimension,
        target_stats=landsat_stats
    )
    if overview_image is None: return {"error": "Failed to fetch overview"}

    output_path = Path(OUTPUT_DIR) / "detections"
    output_path.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overview_image).save(output_path / "overview_raw.jpg")

    # 2. Run Global Segmentation
    print("  Running Global Segmentation...")
    detector = MiningSegmentationDetector(model_name="wu-pr-gw/segformer-b2-finetuned-with-LoveDA")
    visualizer = DetectionVisualizer()

    # Get Masks
    raw_mask = detector.predict_mask(overview_image)
    suspicious_mask = detector.get_suspicious_mask(raw_mask, smooth=True)

    # Save Visualizations
    color_overlay = visualizer.draw_segmentation_overlay(overview_image, raw_mask)
    Image.fromarray(color_overlay).save(output_path / "overview_segmentation_classes.jpg")
    
    suspicious_overlay = visualizer.draw_suspicious_overlay(overview_image, suspicious_mask)
    Image.fromarray(suspicious_overlay).save(output_path / "overview_segmentation_suspicious.jpg")
    print(f"  Saved segmentations to {output_path}")

    # 3. Process Grid
    h, w = overview_image.shape[:2]
    total_width_km = radius_km * 2
    grid_n = max(2, int(total_width_km // 2.5)) 
    step_y, step_x = h // grid_n, w // grid_n
    
    print(f"\n  [Stage 2] Processing {grid_n}x{grid_n} Grid...")
    classifier = Predictor(model_path=str(model_path), backbone=BACKBONE, image_size=MODEL_IMAGE_SIZE)
    results = []
    
    viz_img = Image.fromarray(overview_image).convert("RGBA")
    draw = ImageDraw.Draw(viz_img)
    bbox = center_to_bbox(center_lat, center_lon, radius_km)
    
    classification_dir = output_path / "classification_crops"
    classification_dir.mkdir(parents=True, exist_ok=True)

    for r in range(grid_n):
        for c in range(grid_n):
            y1, x1 = r * step_y, c * step_x
            y2 = h if r == grid_n - 1 else (r + 1) * step_y
            x2 = w if c == grid_n - 1 else (c + 1) * step_x
            
            # Analyze Mask
            cell_mask = suspicious_mask[y1:y2, x1:x2]
            suspicious_pixels = np.count_nonzero(cell_mask)
            suspicious_ratio = suspicious_pixels / cell_mask.size
            
            # Draw Grid
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 255, 80), width=1)
            
            # DEBUG PRINT
            # print(f"    Grid ({r},{c}): Ratio {suspicious_ratio:.1%}")

            # Threshold > 5%
            if suspicious_ratio > 0.05:
                cy_px, cx_px = (y1 + y2) // 2, (x1 + x2) // 2
                cell_lat = bbox[3] - (cy_px / h) * (bbox[3] - bbox[1])
                cell_lon = bbox[0] + (cx_px / w) * (bbox[2] - bbox[0])
                
                rgb, _ = fetcher.fetch_image(
                    lat=cell_lat, lon=cell_lon,
                    distance_km=classification_distance_km,
                    target_stats=landsat_stats,
                    simulate_landsat=True
                )
                
                if rgb is not None:
                    # Save Crop
                    crop_filename = f"cell_{r}_{c}_{suspicious_ratio:.2f}.jpg"
                    crop_path = classification_dir / crop_filename
                    Image.fromarray(rgb).save(crop_path)
                    
                    # Classify
                    pred = classifier.predict(str(crop_path))
                    
                    if pred["is_mining"]:
                        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=4)
                        results.append({"lat": cell_lat, "lon": cell_lon, "prob": pred["probability"], "type": "mining"})
                        print(f"    Cell ({r},{c}): 🚨 MINING ({pred['probability']:.1%}) - Suspicious Area: {suspicious_ratio:.1%}")
                    else:
                        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                        print(f"    Cell ({r},{c}): Clean ({pred['probability']:.1%}) - Suspicious Area: {suspicious_ratio:.1%}")
                else:
                    print(f"    Cell ({r},{c}): Fetch Failed")
            else:
                pass # Skip

    viz_img.convert("RGB").save(output_path / "grid_analysis.jpg")
    print(f"\n  Saved visualization to {output_path / 'grid_analysis.jpg'}")
    
    return {"results": results}




# =============================================================================
# STEP 6: Validate Model Against Known Coordinates
# =============================================================================

def step6_validate(
    model_dir: str,
    validation_csv: Optional[str] = None,
    known_coordinates: Optional[List[Dict]] = None,
    sample_size: Optional[int] = None
) -> Dict:
    print("\n" + "=" * 60)
    print("STEP 6: Validating Model Against Known Coordinates")
    print("=" * 60)
    
    model_path = Path(model_dir) / "best_model.pth"
    
    if not model_path.exists():
        print(f"  Error: Model not found at {model_path}")
        return {"error": "Model not found"}
    
    coordinates = []
    
    if known_coordinates:
        coordinates = known_coordinates
    elif validation_csv and Path(validation_csv).exists():
        print(f"  Loading validation data from: {validation_csv}")
        with open(validation_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat_col = next((col for col in row.keys() if col.lower() in ['lat', 'latitude']), None)
                    lon_col = next((col for col in row.keys() if col.lower() in ['lon', 'lng', 'longitude']), None)
                    label_col = next((col for col in row.keys() if col.lower() in ['label', 'class', 'type']), None)
                    
                    if lat_col and lon_col and label_col:
                        coordinates.append({
                            "lat": float(row[lat_col]),
                            "lon": float(row[lon_col]),
                            "label": row[label_col].lower()
                        })
                except (ValueError, KeyError):
                    continue
    
    if not coordinates:
        print("  No validation coordinates provided. Skipping validation.")
        return {"skipped": True}
    
    if sample_size and len(coordinates) > sample_size:
        import random
        coordinates = random.sample(coordinates, sample_size)
    
    print(f"  Validating on {len(coordinates)} samples")
    
    output_path = Path(OUTPUT_DIR) / "validation"
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from satellite_fetcher import SatelliteFetcher
        from PIL import Image
    except ImportError as e:
        print(f"  Error: {e}")
        return {"error": str(e)}
    
    fetcher = SatelliteFetcher(date_range=DATE_RANGE, max_cloud_cover=MAX_CLOUD_COVER)
    classifier = Predictor(model_path=str(model_path), backbone=BACKBONE, image_size=MODEL_IMAGE_SIZE, threshold=MINING_THRESHOLD)
    
    results = []
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for i, coord in enumerate(coordinates):
        rgb, metadata = fetcher.fetch_image(lat=coord["lat"], lon=coord["lon"], distance_km=IMAGE_SIZE_KM)
        
        if rgb is None:
            results.append({"lat": coord["lat"], "lon": coord["lon"], "known_label": coord["label"], "predicted": None, "correct": None, "error": metadata.get("error")})
            continue
        
        img_path = images_dir / f"val_{i:04d}_{coord['label']}.jpg"
        Image.fromarray(rgb).save(img_path, quality=95)
        
        # Free rgb after saving
        del rgb
        
        pred = classifier.predict(str(img_path))
        predicted_label = "mining" if pred["is_mining"] else "forest"
        known_is_mining = coord["label"] in ["mining", "mine", "positive", "1", "true"]
        
        correct = (pred["is_mining"] == known_is_mining)
        
        if known_is_mining and pred["is_mining"]: tp += 1
        elif known_is_mining and not pred["is_mining"]: fn += 1
        elif not known_is_mining and pred["is_mining"]: fp += 1
        else: tn += 1
        
        results.append({"lat": coord["lat"], "lon": coord["lon"], "known_label": coord["label"], "predicted": predicted_label, "probability": pred["probability"], "correct": correct, "image_path": str(img_path)})
        
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(coordinates)}")
            gc.collect()
        
        time.sleep(0.5)
    
    total = tp + tn + fp + fn
    metrics = {
        "total": len(coordinates), "processed": total,
        "accuracy": (tp + tn) / total if total > 0 else 0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "f1": 2 * (tp / (tp + fp)) * (tp / (tp + fn)) / ((tp / (tp + fp)) + (tp / (tp + fn))) if (tp + fp) > 0 and (tp + fn) > 0 else 0,
        "true_positives": tp, "false_positives": fp, "true_negatives": tn, "false_negatives": fn
    }
    
    print(f"\n  VALIDATION RESULTS")
    print(f"  Accuracy: {metrics['accuracy']:.1%}, Precision: {metrics['precision']:.1%}, Recall: {metrics['recall']:.1%}, F1: {metrics['f1']:.3f}")
    
    output = {"metrics": metrics, "results": results}
    
    results_path = output_path / "validation_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    
    return output


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main(
    skip_data_collection: bool = False,
    skip_training: bool = False,
    center_lat: Optional[float] = None,
    center_lon: Optional[float] = None,
    radius_km: float = OVERVIEW_RADIUS_KM,
    overview_max_dimension: int = 2048,
    classification_distance_km: float = 2.5,
    validation_csv: Optional[str] = None,
    known_coordinates: Optional[List[Dict]] = None
):
    print("\n" + "=" * 60)
    print("ILLEGAL MINING DETECTION PIPELINE")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    outputs = {}
    model_dir = Path(OUTPUT_DIR) / "model"
    landsat_stats = None
    
    if not skip_training:
        if not skip_data_collection:
            outputs["step1"] = step1_collect_mines()
            landsat_stats = outputs["step1"]["landsat_stats"]
            outputs["step2"] = step2_collect_forest(landsat_stats=landsat_stats)
            outputs["step3"] = step3_build_dataset(
                mines_dir=outputs["step1"]["output_dir"],
                forest_dir=outputs["step2"]["output_dir"],
                landsat_stats=landsat_stats
            )
        else:
            print("Skipping data collection (steps 1-2)")
            outputs["step3"] = {"output_dir": str(Path(OUTPUT_DIR) / "dataset")}
            stats_file = Path(OUTPUT_DIR) / "landsat_stats.json"
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    landsat_stats = json.load(f)
        
        outputs["step4"] = step4_train(dataset_dir=outputs["step3"]["output_dir"])
        model_dir = outputs["step4"]["model_dir"]
    else:
        print("Skipping training (steps 1-4), using existing model")
        if not (Path(model_dir) / "best_model.pth").exists():
            print(f"  ERROR: No model found at {model_dir}/best_model.pth")
            return outputs
        
        stats_file = Path(OUTPUT_DIR) / "landsat_stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                landsat_stats = json.load(f)
    
    outputs["step5"] = step5_detect_overview(
        model_dir=str(model_dir),
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=radius_km,
        overview_max_dimension=overview_max_dimension,
        classification_distance_km=classification_distance_km,
        landsat_stats=landsat_stats
    )
    
    outputs["step6"] = step6_validate(
        model_dir=str(model_dir),
        validation_csv=validation_csv or VALIDATION_CSV,
        known_coordinates=known_coordinates,
    )
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    if outputs.get("step5") and not outputs["step5"].get("skipped"):
        n_mining = outputs["step5"].get("n_confirmed_mining", 0)
        print(f"  Detections: {n_mining} mining zones detected")
    
    return outputs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Illegal Mining Detection Pipeline")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, use existing model")
    parser.add_argument("--skip-data", action="store_true", help="Skip data collection")
    parser.add_argument("--center", type=str, help="Center coordinates: LAT,LON (e.g., -14.2,-49.4)")
    parser.add_argument("--radius", type=float, default=OVERVIEW_RADIUS_KM, help="Radius from center in km (default: 10)")
    parser.add_argument("--overview-size", type=int, default=2048, help="Max dimension for overview (px)")
    parser.add_argument("--classification-distance", type=float, default=2.5, help="Distance for classification crops (km)")
    parser.add_argument("--validate", type=str, help="Path to validation CSV")
    
    args = parser.parse_args()
    
    center_lat, center_lon = None, None
    if args.center:
        parts = args.center.split(",")
        if len(parts) == 2:
            center_lat = float(parts[0])
            center_lon = float(parts[1])
        else:
            print("Error: --center must be LAT,LON (e.g., -14.2,-49.4)")
            exit(1)
    

    # python3 main.py --skip-training --center="-14.2,-49.4" --radius 10 --overview-size 2048
    main(
        skip_training=args.skip_training,
        skip_data_collection=args.skip_data,
        center_lat=center_lat,
        center_lon=center_lon,
        radius_km=args.radius,
        overview_max_dimension=args.overview_size,
        classification_distance_km=args.classification_distance,
        validation_csv=args.validate
    )