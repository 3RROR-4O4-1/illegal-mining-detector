### `NOTEBOOK_docs.pdf` <- Notebook PDF
### `notebook.ipynb` <- Executable Notebook
### `Presentation.pdf` <- Project Presentation




# A- System dependencies
```bash
sudo apt-get update
sudo apt-get install -y \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    python3-dev \
    build-essential

```

**Note: these commands are for Linux Ubuntu.  In windows the system dependencies are**

```bash
conda install -c conda-forge rasterio gdal
```

# B - requirements.txt

```bash
pip install requirements.txt
```



# C - Quickstart Execution (or cf main() for the args)

```python
# Full pipeline (training + detection + validation)

python main.py

# Skip training, use existing model
python main.py --skip-training

# With overview image for two-stage detection
python main.py --skip-training --overview amazon_overview.jpg --bounds "-60.5,-4.5,-60.0,-4.0"

# With validation CSV
python main.py --skip-training --validate known_mining_sites.csv
```



<br>

```bash

┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTEGRATED PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: Collect Mine Images (positive samples)                             │
│       ↓                                                                      │
│  STEP 2: Collect Forest Images (negative samples)                           │
│       ↓                                                                      │
│  STEP 3: Build Dataset (augmentation + style matching)                      │
│       ↓                                                                      │
│  STEP 4: Train Classifier (ResNet/EfficientNet)                             │
│       ↓                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 5: Two-Stage Detection                                 │    │
│  │   ├── SegFormer → candidate zones + bounding boxes                  │    │
│  │   ├── Visualizations (boxes, segmentation overlay, grid)            │    │
│  │   ├── Fetch hi-res for each candidate                               │    │
│  │   └── Classifier → confirm mining vs forest                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       ↓                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 6: Validation                                           │    │
│  │   ├── Load known mining coordinates                                 │    │
│  │   ├── Fetch satellite images                                        │    │
│  │   ├── Run predictions                                               │    │
│  │   └── Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│       ↓                                                                      │
│  STEP 7: Batch Inference (simple directory prediction)                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘


```