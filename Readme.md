# Lung Cancer Segmentation and Detection Using YOLOv8 and Hybrid GLCM-CNN Pipeline

This project implements a two-stage deep learning pipeline for automated lung tumor segmentation and detection on medical imaging data. The workflow combines 3D-to-2D data preparation, state-of-the-art segmentation with YOLOv8, extraction of hard negative patches, texture-based **GLCM (Gray-Level Co-occurrence Matrix) feature** computation, and a hybrid convolutional-neural/classifier network for improved prediction accuracy and interpretability.

## Features

- **NIfTI medical image processing:** Converts 3D NIfTI lung CT scans and labels into 2D slices in YOLO segmentation format.
- **YOLOv8 segmentation:** Trains and validates a YOLOv8 segmentation model for accurate tumor boundary detection.
- **Comprehensive visualization:** Produces side-by-side animations for original, ground-truth, and YOLO-predicted masks, both for validation and test datasets.
- **False positive extraction:** Identifies both true tumor regions and YOLOv8 model's common false positives, cropping patches for secondary classifier training.
- **GLCM texture feature extraction:** Computes Gray-Level Co-occurrence Matrix (GLCM)-based texture features (contrast, dissimilarity, homogeneity, energy, correlation) for each tumor or false positive patch.
- **Hybrid CNN-GLCM analyzer:** Trains a PyTorch model that fuses texture features and deep CNN features (from pre-trained ResNet50) for robust tumor vs healthy tissue classification.
- **End-to-end pipeline demonstration:** Renders an animation overlaying YOLOv8 + hybrid analyzer predictions on raw CT slices.

---

## Project Structure

```
lung-cancer-seg/
├── data/                # Not included: expects Decathlon Lung dataset structure (3D NIfTI files)
├── yolo_data/           # Auto-generated: YOLO-format dataset after conversion
├── stage2_analyzer/     # Auto-generated: Class 0 healthy, Class 1 cancer patch datasets
├── runs/                # YOLO training outputs, weights, and logs
├── *.py / *.ipynb       # Python scripts / notebooks for each stage (see below)
└── README.md
```

---

## Requirements

- Python ≥ 3.8
- pip install: `ultralytics nibabel scikit-image opencv-python-headless tqdm torch torchvision`
- Optionally: `IPython`, Jupyter Notebook/Lab for visualization, running in Kaggle/Colab or local environment

---

## Workflow Overview

### 1. Data Preparation and YOLOv8 Formatting

- Converts 3D NIfTI scans and segmentation masks to:
   - 2D slices (PNG)
   - YOLO polygon format label files
- Training/validation/test split management
- Generates YOLO `data.yaml` configuration for one-class segmentation ('tumor')

### 2. YOLOv8 Training and Validation

- Loads/instantiates `yolov8n-seg.pt` or another YOLOv8 segmentation model
- Trains on prepared dataset (custom epochs/img size/batch size adjustable)
- Saves weights and validation results to `runs/segment/*`

### 3. Visualization

- Generates videos/animations of:
   - Original slice
   - Ground truth mask overlay
   - Model prediction mask overlay

### 4. Stage 2 Patch Extraction

- Extracts:
   - Real tumor patches (ground truth)
   - False positive patches (model mistakes)
- Stores them in `stage2_analyzer/class_0_healthy/` and `stage2_analyzer/class_1_cancer/`

### 5. GLCM Texture Feature Extraction

- Computes five GLCM-based texture features for each patch

### 6. Hybrid Analyzer Model (PyTorch)

- Defines a PyTorch dataset and architecture:
   - ResNet50 base (frozen, with a new FC layer)
   - GLCM feature MLP path
   - Late fusion and sigmoid for classification
- Trains and evaluates using cross-entropy loss and accuracy

### 7. Full Smart Analyzer Pipeline & Animation

- For each test image slice:
   - Runs YOLOv8 segmentation
   - For each detection, extracts the patch, computes GLCM features, and classifies with the hybrid model
   - Renders results side-by-side

---

## Usage and Examples

_**Note:** Replace file paths as needed for your environment. The pipeline assumes the [Medical Segmentation Decathlon Lung dataset](http://medicaldecathlon.com/) directory format and NIfTI files._

### Quickstart

**1. Install requirements:**
```bash
pip install ultralytics nibabel scikit-image opencv-python-headless tqdm torch torchvision
```

**2. Prepare data structure:**
```
/your-dataset-root/
    imagesTr/      # Training images (3D NIfTI)
    labelsTr/      # Training labels (3D NIfTI)
    imagesTs/      # Test images (3D NIfTI)
```

**3. Run scripts sequentially (or in a notebook):**
- **Data conversion and YOLO training**  
   Prepare 2D data, labels, and train YOLOv8 segmentation.
- **Visualization**  
   Use provided code to generate prediction videos/animations.
- **Patch extraction & GLCM feature calculation**  
   (Stage 2) Generate real tumor and false positive patches, extract GLCM features.
- **Train Hybrid Analyzer**  
   Train the PyTorch CNN+GLCM classifier.
- **Final animation**  
   Visualize model predictions on new test scans.

**4. Example: Data conversion function**
```python
convert_nii_to_yolo(image_path, label_path, output_img_dir, output_lbl_dir, patient_id)
```
(Sample invocations provided in scripts above)

**5. Training YOLOv8 Model**
```python
from ultralytics import YOLO
model = YOLO('yolov8n-seg.pt')
model.train(data='path/to/data.yaml', epochs=50, imgsz=512, batch=16, name='yolov8_lung_seg')
```

**6. Training Hybrid CNN+GLCM Analyzer**
```python
# See scripts above for full training loop and dataset setup
analyzer_model = HybridAnalyzer()
analyzer_model.train(...)
```

**7. Visualization / Animation**
- See video writing sections in code; videos saved to `/kaggle/working/` or equivalent.

---

## Notes

- This project requires significant disk and compute resources (GPU recommended).
- All code should be runnable both as standalone scripts or through Jupyter/Kaggle notebooks; adapt paths as needed.
- By default, no user, license, or dataset links are included. Please add credits or licensing information as appropriate for your use case.

---

## Acknowledgements

Developed as a 7th semester student project on cancer detection and segmentation using segmentation and hybrid deep learning architectures.

---

## Contact

For queries or discussion, please open an issue or contact the maintainer via GitHub.
