"""
Satellite image fetching from Microsoft Planetary Computer.
Uses Sentinel-2 but simulates Landsat-upscaled aesthetics to match local training data.

Optimized for memory efficiency with large images.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import json
import time
from typing import Optional, Tuple, Dict, List
import random
import gc

import planetary_computer as pc
from pystac_client import Client
import rasterio
from rasterio.warp import transform_bounds

PROTECTED_AREAS = [
    {"name": "Tumucumaque", "lat": 1.5, "lon": -52.5, "radius_km": 100},
    {"name": "Jaú", "lat": -2.0, "lon": -63.0, "radius_km": 80},
    {"name": "Mamirauá", "lat": -2.5, "lon": -65.0, "radius_km": 50},
    {"name": "Terra do Meio", "lat": -5.5, "lon": -53.0, "radius_km": 100},
    {"name": "Xingu", "lat": -10.5, "lon": -52.5, "radius_km": 100},
]


def center_to_bbox(
    center_lat: float,
    center_lon: float,
    radius_km: float
) -> Tuple[float, float, float, float]:
    """
    Convert center coordinate + radius to bounding box.
    
    Args:
        center_lat: Latitude of center point
        center_lon: Longitude of center point  
        radius_km: Distance from center to edge in km
        
    Returns:
        (lon_min, lat_min, lon_max, lat_max)
    """
    # Latitude: 1 degree ≈ 111 km
    lat_offset = radius_km / 111.0
    
    # Longitude: depends on latitude (narrower near poles)
    lon_offset = radius_km / (111.0 * np.cos(np.radians(center_lat)))
    
    return (
        center_lon - lon_offset,  # lon_min
        center_lat - lat_offset,  # lat_min
        center_lon + lon_offset,  # lon_max
        center_lat + lat_offset   # lat_max
    )


class SatelliteFetcher:
    def __init__(
        self,
        date_range: str = "2023-01-01/2024-12-31",
        max_cloud_cover: int = 20
    ):
        self.collection = "sentinel-2-l2a"
        self.date_range = date_range
        self.max_cloud_cover = max_cloud_cover

        self.client = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace
        )
        self.bands = ["B04", "B03", "B02"]
    
    def fetch_image(
        self,
        lat: float,
        lon: float,
        distance_km: float = 2.5,
        target_stats: Optional[Dict] = None,
        output_size: int = 512,
        simulate_landsat: bool = True
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Fetch a satellite image centered on a point.
        
        Args:
            lat: Latitude of center point
            lon: Longitude of center point
            distance_km: Distance from center to edge in km
            target_stats: Color statistics to match (optional)
            output_size: Output image size in pixels
            simulate_landsat: If True, applies blur to match Landsat-upscaled quality
            
        Returns:
            (RGB image as numpy array, metadata dict)
        """
        bbox = center_to_bbox(lat, lon, distance_km)
        
        try:
            search = self.client.search(
                collections=[self.collection],
                bbox=bbox,
                datetime=self.date_range,
                query={"eo:cloud_cover": {"lt": self.max_cloud_cover}}
            )
            
            items = list(search.items())
            if not items:
                return None, {"error": "No imagery found", "lat": lat, "lon": lon}
            
            items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
            item = items[0]
            
            rgb_bands = []
            for band_name in self.bands:
                href = item.assets[band_name].href
                with rasterio.open(href) as src:
                    src_bbox = transform_bounds('EPSG:4326', src.crs, *bbox)
                    window = src.window(*src_bbox)
                    data = src.read(1, window=window)
                    if data.size == 0:
                        return None, {"error": "Empty", "lat": lat, "lon": lon}
                    rgb_bands.append(data.astype(np.float32))
            
            # Align shapes
            min_h = min(b.shape[0] for b in rgb_bands)
            min_w = min(b.shape[1] for b in rgb_bands)
            rgb_bands = [b[:min_h, :min_w] for b in rgb_bands]

            rgb = np.stack(rgb_bands, axis=-1)
            del rgb_bands  # Free memory
            
            # Normalization (in-place where possible)
            np.clip(rgb, 0, 10000, out=rgb)
            rgb /= 10000.0
            rgb_enhanced = self._enhance_contrast(rgb)
            del rgb
            
            if target_stats:
                rgb_final = self._normalize_to_target(rgb_enhanced, target_stats)
                del rgb_enhanced
            else:
                rgb_final = rgb_enhanced
            
            rgb_uint8 = (np.clip(rgb_final, 0, 1) * 255).astype(np.uint8)
            del rgb_final
            
            img = Image.fromarray(rgb_uint8)
            del rgb_uint8
            
            if simulate_landsat:
                low_res_size = 93 
                img_small = img.resize((low_res_size, low_res_size), Image.BILINEAR)
                img = img_small.resize((output_size, output_size), Image.BICUBIC)
                platform_note = "Sentinel-2 (Downsampled to match Landsat)"
            else:
                img = img.resize((output_size, output_size), Image.LANCZOS)
                platform_note = "Sentinel-2 (Full Resolution)"
            
            metadata = {
                "lat": lat, "lon": lon,
                "platform": platform_note,
                "cloud_cover": item.properties.get("eo:cloud_cover", None),
                "datetime": item.properties.get("datetime", None)
            }
            
            return np.array(img), metadata
            
        except Exception as e:
            return None, {"error": str(e), "lat": lat, "lon": lon}
    
    def fetch_overview(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float = 10.0,
        max_dimension: int = 2048,
        target_stats: Optional[Dict] = None
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Fetch a HIGH-RESOLUTION overview image centered on a point.
        
        This method fetches Sentinel-2 imagery at full resolution (no blur simulation)
        for use with SegFormer segmentation.
        
        Args:
            center_lat: Latitude of center point
            center_lon: Longitude of center point
            radius_km: Distance from center to edge in km (default 10km = 20km x 20km area)
            max_dimension: Maximum width or height of output image
            target_stats: Color statistics to match (optional)
            
        Returns:
            (RGB image as numpy array, metadata dict with bounds info)
        """
        bbox = center_to_bbox(center_lat, center_lon, radius_km)
        lon_min, lat_min, lon_max, lat_max = bbox
        
        lat_span_km = (lat_max - lat_min) * 111.0
        lon_span_km = (lon_max - lon_min) * 111.0 * np.cos(np.radians(center_lat))
        
        print(f"    Fetching overview: {lon_span_km:.1f} km x {lat_span_km:.1f} km")
        print(f"    Center: ({center_lat:.4f}, {center_lon:.4f}), Radius: {radius_km} km")
        
        try:
            search = self.client.search(
                collections=[self.collection],
                bbox=list(bbox),
                datetime=self.date_range,
                query={"eo:cloud_cover": {"lt": self.max_cloud_cover}}
            )
            
            items = list(search.items())
            if not items:
                return None, {"error": "No imagery found", "bbox": bbox}
            
            items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 100))
            item = items[0]
            
            print(f"    Found {len(items)} images, using best with {item.properties.get('eo:cloud_cover', '?')}% cloud")
            
            # Load all bands and determine actual shapes from data
            rgb_bands = []
            native_shape = None
            
            for band_name in self.bands:
                href = item.assets[band_name].href
                with rasterio.open(href) as src:
                    src_bbox = transform_bounds('EPSG:4326', src.crs, *bbox)
                    window = src.window(*src_bbox)
                    data = src.read(1, window=window)
                    
                    if data.size == 0:
                        return None, {"error": "Empty window", "bbox": bbox}
                    
                    if native_shape is None:
                        native_shape = data.shape
                        print(f"    Native resolution: {data.shape[1]}x{data.shape[0]} pixels")
                    
                    rgb_bands.append(data.astype(np.float32))
                    del data
                
                gc.collect()
            
            # Use minimum dimensions across bands (they can differ slightly)
            min_h = min(b.shape[0] for b in rgb_bands)
            min_w = min(b.shape[1] for b in rgb_bands)
            
            # Stack into single array, trimming to common dimensions
            rgb = np.zeros((min_h, min_w, 3), dtype=np.float32)
            for i, band in enumerate(rgb_bands):
                rgb[:, :, i] = band[:min_h, :min_w]
            
            del rgb_bands
            gc.collect()
            
            # In-place normalization to save memory
            np.clip(rgb, 0, 10000, out=rgb)
            rgb /= 10000.0
            
            # Enhance contrast (returns new array, but we delete old immediately)
            rgb_enhanced = self._enhance_contrast(rgb)
            del rgb
            gc.collect()
            
            if target_stats:
                rgb_final = self._normalize_to_target(rgb_enhanced, target_stats)
                del rgb_enhanced
            else:
                rgb_final = rgb_enhanced
            
            # Convert to uint8 (final output format)
            rgb_uint8 = (np.clip(rgb_final, 0, 1) * 255).astype(np.uint8)
            del rgb_final
            gc.collect()
            
            # Resize if needed
            h, w = rgb_uint8.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = Image.fromarray(rgb_uint8)
                del rgb_uint8
                img = img.resize((new_w, new_h), Image.LANCZOS)
                rgb_uint8 = np.array(img)
                del img
                print(f"    Resized to: {new_w}x{new_h} pixels")
            
            final_h, final_w = rgb_uint8.shape[:2]
            meters_per_pixel_x = (lon_span_km * 1000) / final_w
            meters_per_pixel_y = (lat_span_km * 1000) / final_h
            
            metadata = {
                "bbox": bbox,
                "center_lat": center_lat,
                "center_lon": center_lon,
                "radius_km": radius_km,
                "width_km": lon_span_km,
                "height_km": lat_span_km,
                "image_width": final_w,
                "image_height": final_h,
                "meters_per_pixel": (meters_per_pixel_x + meters_per_pixel_y) / 2,
                "platform": "Sentinel-2 (Full Resolution)",
                "cloud_cover": item.properties.get("eo:cloud_cover", None),
                "datetime": item.properties.get("datetime", None),
                "native_resolution": native_shape
            }
            
            return rgb_uint8, metadata
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, {"error": str(e), "bbox": bbox}
    
    def _enhance_contrast(self, rgb: np.ndarray) -> np.ndarray:
        """Enhance contrast using percentile stretching."""
        result = np.zeros_like(rgb, dtype=np.float32)
        for i in range(3):
            band = rgb[:, :, i]
            valid = band[band > 0.001]
            if len(valid) > 0:
                p2, p98 = np.percentile(valid, [2, 98])
                if p98 > p2:
                    result[:, :, i] = (band - p2) / (p98 - p2)
                else:
                    result[:, :, i] = band
            else:
                result[:, :, i] = band
        return np.clip(result, 0, 1)
    
    def _normalize_to_target(self, rgb: np.ndarray, target_stats: Dict) -> np.ndarray:
        """Normalize colors to match target statistics."""
        result = np.zeros_like(rgb, dtype=np.float32)
        for i, channel in enumerate(["r", "g", "b"]):
            band = rgb[:, :, i]
            valid = band[band > 0.001]
            if len(valid) > 0:
                src_p2, src_p98 = np.percentile(valid, [2, 98])
                tgt_p2 = target_stats.get(f"{channel}_p2", 0.0)
                tgt_p98 = target_stats.get(f"{channel}_p98", 1.0)
                if src_p98 > src_p2:
                    normalized = (band - src_p2) / (src_p98 - src_p2)
                    result[:, :, i] = normalized * (tgt_p98 - tgt_p2) + tgt_p2
                else:
                    result[:, :, i] = band
            else:
                result[:, :, i] = band
        return np.clip(result, 0, 1)
    
    def generate_forest_samples(
        self,
        n_samples: int,
        output_dir: str,
        target_stats: Optional[Dict] = None,
        brazil_bounds: Tuple[float, float, float, float] = (-75, -35, -35, 5),
        distance_km: float = 2.5
    ) -> List[Dict]:
        """Generate random forest samples from protected areas."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        samples = []
        attempts = 0
        max_attempts = n_samples * 10 
        
        print(f"    Targeting {n_samples} samples (simulating upscaled quality)...")
        
        while len(samples) < n_samples and attempts < max_attempts:
            attempts += 1
            area = random.choice(PROTECTED_AREAS)
            offset_km = random.uniform(0, area["radius_km"])
            angle = random.uniform(0, 2 * np.pi)
            lat = area["lat"] + (offset_km / 111) * np.sin(angle)
            lon = area["lon"] + (offset_km / 111) * np.cos(angle)
            
            rgb, metadata = self.fetch_image(lat, lon, distance_km=distance_km, target_stats=target_stats)
            
            if rgb is not None:
                mean_val = rgb.mean()
                if 20 < mean_val < 230:
                    filename = f"forest_{len(samples):04d}.jpg"
                    filepath = output_path / filename
                    Image.fromarray(rgb).save(filepath, quality=95)
                    metadata["filename"] = filename
                    samples.append(metadata)
                    if len(samples) % 10 == 0:
                        print(f"    [+] Generated {len(samples)}/{n_samples} forest samples")
                
                del rgb
                gc.collect()
        
        with open(output_path / "metadata.json", "w") as f:
            json.dump(samples, f, indent=2)
        return samples