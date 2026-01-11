"""
CNN classifier for mining detection.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    from torchvision import transforms, models
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. Training disabled.")


class MiningDataset(Dataset):
    """
    PyTorch Dataset for mining classification.
    Loads from a specific split directory ('train' or 'val').
    """
    
    def __init__(
        self,
        dataset_dir: str,
        split: str = "train",  # Added split argument
        image_size: int = 224
    ):
        self.image_size = image_size
        self.split = split
        
        # Standard normalization for both train and val
        # (Augmentation is now done offline in build_dataset)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        root_path = Path(dataset_dir) / split
        if not root_path.exists():
            raise ValueError(f"Dataset split directory not found: {root_path}")
            
        self.samples = []
        
        # Load positive samples (Label: 1)
        pos_dir = root_path / "positive"
        if pos_dir.exists():
            for img_file in pos_dir.glob("*.jpg"):
                self.samples.append((str(img_file), 1.0))
        
        # Load negative samples (Label: 0)
        neg_dir = root_path / "negative"
        if neg_dir.exists():
            for img_file in neg_dir.glob("*.jpg"):
                self.samples.append((str(img_file), 0.0))
                
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {root_path}")
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)
            return img_tensor, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy tensor in case of corruption, or handle appropriately
            return torch.zeros((3, self.image_size, self.image_size)), label


class MiningClassifier(nn.Module):
    """CNN classifier with pretrained backbone."""
    
    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 1)
            )
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 1)
            )
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 1)
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def forward(self, x):
        return torch.sigmoid(self.backbone(x))


def train_model(
    dataset_dir: str,
    output_dir: str,
    backbone: str = "resnet18",
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    epochs: int = 30,
    patience: int = 7,
    image_size: int = 224
) -> Dict:
    """
    Train a mining classifier using pre-split folders.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for training")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    
    # 1. Load Datasets (Physical Split)
    print("  Loading Training Set...")
    train_dataset = MiningDataset(dataset_dir, split="train", image_size=image_size)
    
    print("  Loading Validation Set...")
    val_dataset = MiningDataset(dataset_dir, split="val", image_size=image_size)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples:   {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Model Setup
    model = MiningClassifier(backbone=backbone).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # 3. Training Loop
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_dataset)
        
        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_dataset)
        val_acc = correct / total
        
        # Scheduler Step
        scheduler.step(val_loss)
        
        # Logging
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"  Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), output_path / "best_model.pth")
            # print("    -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch+1}")
                break
    
    # Save Final
    torch.save(model.state_dict(), output_path / "final_model.pth")
    
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    return history



class Predictor:
    """Run inference with a trained model."""
    
    def __init__(
        self,
        model_path: str,
        backbone: str = "resnet18",
        image_size: int = 224,
        threshold: float = 0.5
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch required for inference")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.image_size = image_size
        
        # Load model
        self.model = MiningClassifier(backbone=backbone, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path: str) -> Dict:
        """
        Predict on a single image.
        
        Returns:
            {"probability": float, "prediction": str, "is_mining": bool}
        """
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob = self.model(img_tensor).item()
        
        return {
            "probability": prob,
            "prediction": "mining" if prob > self.threshold else "forest",
            "is_mining": prob > self.threshold
        }
    
    def predict_array(self, image: np.ndarray) -> Dict:
        """Predict on a numpy array (RGB, 0-255)."""
        img = Image.fromarray(image)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prob = self.model(img_tensor).item()
        
        return {
            "probability": prob,
            "prediction": "mining" if prob > self.threshold else "forest",
            "is_mining": prob > self.threshold
        }
    
    def predict_batch(self, image_dir: str) -> List[Dict]:
        """Predict on all images in a directory."""
        results = []
        image_path = Path(image_dir)
        
        for img_file in list(image_path.glob("*.jpg")) + list(image_path.glob("*.png")):
            result = self.predict(str(img_file))
            result["filename"] = img_file.name
            results.append(result)
        
        return results
