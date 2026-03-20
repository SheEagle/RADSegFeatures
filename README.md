
# RADSeg Dense Feature Extractor

A minimal script to extract and save dense semantic features from images using the RADSeg foundation model. It outputs a `.pt` file containing both 4D spatial tensors and 2D flattened tensors.

## Usage

### 1. Extract and Save Features

```bash
python extract_dense_features.py football.png
```

### 2. Load and Use Features

```python
import torch

# Load the saved feature dictionary
features = torch.load("image_features.pt")

# Access 4D spatial tensors (Shape: 1, C, H, W)
scga_spatial = features['scga_feat']
aligned_spatial = features['visual_aligned']

# Access 2D flattened tensors (Shape: H*W, C)
scga_flat = features['scga_feat_2d']
aligned_flat = features['visual_aligned_2d']
```

## Output Data Structure

The generated `*_features.pt` file contains a dictionary with the following keys:

| Key | Shape | Description |
| :--- | :--- | :--- |
| `scga_feat` | `(1, C, H, W)` | Base dense features directly from the SCGA encoder. |
| `scga_feat_2d` | `(H*W, C)` | Flattened version of the base SCGA features. |
| `visual_aligned` | `(1, C, H, W)` | Dense features mapped to the language semantic space. |
| `visual_aligned_2d`| `(H*W, C)` | Flattened version of the language-aligned features. |


