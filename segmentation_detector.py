import numpy as np
import cv2
import gc
from typing import Optional, List, Tuple, Dict

try:
    import torch
    from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
    HAS_SEGFORMER = True
except ImportError:
    HAS_SEGFORMER = False

class MiningSegmentationDetector:
    """
    Robust SegFormer detector.
    Identifies 'suspicious' pixels including industrial/disturbed land.
    """
    
    # LoveDA Class Labels:
    # 0:Background, 1:Building, 2:Road, 3:Water, 4:Barren, 5:Forest, 6:Agricultural
    
    def __init__(
        self,
        model_name: str = "wu-pr-gw/segformer-b2-finetuned-with-LoveDA",
        device: Optional[str] = None
    ):
        if not HAS_SEGFORMER:
            raise ImportError("transformers required: pip install transformers")
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SegFormer: {model_name}")
        
        try:
            self.processor = SegformerImageProcessor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        except Exception as e:
            print(f"Warning: {e}. Fallback to generic model.")
            fallback = "nvidia/segformer-b0-finetuned-ade-512-512"
            self.processor = SegformerImageProcessor.from_pretrained(fallback)
            self.model = SegformerForSemanticSegmentation.from_pretrained(fallback)

        self.model.to(self.device)
        self.model.eval()
        
       # clsses are Building, Road, Water, Barren
        self.suspicious_classes = {1, 2, 3, 4} 

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        """Returns the raw class ID mask (H, W)."""
        h, w = image.shape[:2]
        
        max_dim = 2048
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_input = cv2.resize(image, (new_w, new_h))
        else:
            img_input = image

        inputs = self.processor(images=img_input, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = torch.nn.functional.interpolate(
                outputs.logits, size=(h, w), mode="bilinear", align_corners=False
            )
            pred_mask = logits.argmax(dim=1).squeeze().cpu().numpy()
            
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        
        return pred_mask.astype(np.uint8)

    def get_suspicious_mask(self, raw_mask: np.ndarray, smooth: bool = True) -> np.ndarray:
        """
        Converts raw class mask to a binary 'Suspicious' mask.
        """
        suspicious = np.isin(raw_mask, list(self.suspicious_classes)).astype(np.uint8) * 255
        
        if not smooth:
            return suspicious

        # BALANCED SMOOTHING (5x5)
        # Big enough to remove salt-and-pepper noise.
        # Small enough to keep mines distinct from towns.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.morphologyEx(suspicious, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_OPEN, kernel)
        
        return smoothed

class DetectionVisualizer:
    CLASS_COLORS = np.array([
        [0, 0, 0],       # 0: Background
        [255, 0, 0],     # 1: Building (Red)
        [255, 215, 0],   # 2: Road (Gold)
        [0, 0, 255],     # 3: Water (Blue)
        [160, 82, 45],   # 4: Barren (Brown)
        [34, 139, 34],   # 5: Forest (Green)
        [154, 205, 50]   # 6: Agri (LtGreen)
    ], dtype=np.uint8)

    def draw_segmentation_overlay(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        safe_mask = np.clip(mask, 0, len(self.CLASS_COLORS) - 1)
        colored_mask = self.CLASS_COLORS[safe_mask]
        return (image * (1 - alpha) + colored_mask * alpha).astype(np.uint8)
    
    def draw_suspicious_overlay(self, image: np.ndarray, binary_mask: np.ndarray) -> np.ndarray:
        overlay = image.copy()
        overlay[binary_mask > 0] = [255, 0, 0] # Red highlight
        return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)