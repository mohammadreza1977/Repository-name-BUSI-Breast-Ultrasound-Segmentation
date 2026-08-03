# BUSI Breast Ultrasound Image Segmentation using U-Net

A deep learning project for **breast ultrasound image segmentation** using a U-Net architecture implemented with PyTorch.

The project uses the BUSI breast ultrasound dataset and provides a complete pipeline for dataset inspection, preprocessing, train/validation/test splitting, model training, and quantitative evaluation.

## Project Overview

The objective of this project is to automatically segment breast lesions from ultrasound images.

The implemented pipeline includes:

* Dataset inspection and visualization
* Image and segmentation-mask loading
* Train / validation / test splitting
* U-Net model implementation
* GPU-accelerated training with PyTorch
* Model checkpointing based on validation loss
* Test-set evaluation
* Dice, IoU, Precision, and Recall metrics
* Detailed per-sample evaluation

## Dataset

The project was developed using the **BUSI (Breast Ultrasound Images)** dataset.

Dataset statistics used in this project:

| Category     | Number |
| ------------ | ------ |
| Total images | 629    |
| Benign       | 420    |
| Malignant    | 209    |

Each sample contains:

* An RGB ultrasound image
* A corresponding grayscale segmentation mask

Images and labels are intentionally excluded from this repository through `.gitignore`.

## Dataset Split

The dataset was divided into training, validation, and test sets:

| Split      | Samples |
| ---------- | ------- |
| Training   | 440     |
| Validation | 94      |
| Test       | 95      |
| **Total**  | **629** |

Class distribution:

| Split      | Benign | Malignant |
| ---------- | ------ | --------- |
| Train      | 294    | 146       |
| Validation | 63     | 31        |
| Test       | 63     | 32        |

No missing segmentation labels were found.

## Model

The segmentation model is based on the **U-Net architecture**, a widely used encoder-decoder architecture for biomedical image segmentation.

The implemented model receives:

```text
Input:  3 × 256 × 256
Output: 1 × 256 × 256
```

The model contains approximately:

```text
31,043,521 parameters
```

Training was performed using CUDA on an NVIDIA GeForce RTX 3060 Laptop GPU.

## Training

The model was trained for 30 epochs.

The best model was selected according to the lowest validation loss.

Best checkpoint:

```text
Epoch: 28
Validation Loss: 0.2434
```

The trained checkpoint is intentionally excluded from the repository because of Git repository size considerations.

## Test Results

The final model was evaluated on the held-out test set containing 95 images.

### Overall Results

| Metric        | Mean ± Std          |
| ------------- | ------------------- |
| **Dice**      | **0.7334 ± 0.2450** |
| **IoU**       | **0.6264 ± 0.2493** |
| **Precision** | **0.8089 ± 0.2293** |
| **Recall**    | **0.7536 ± 0.2758** |

### Performance by Class

#### Benign

| Metric    | Mean ± Std      |
| --------- | --------------- |
| Dice      | 0.7566 ± 0.2389 |
| IoU       | 0.6558 ± 0.2501 |
| Precision | 0.7770 ± 0.2201 |
| Recall    | 0.8268 ± 0.2587 |

#### Malignant

| Metric    | Mean ± Std      |
| --------- | --------------- |
| Dice      | 0.6878 ± 0.2503 |
| IoU       | 0.5685 ± 0.2374 |
| Precision | 0.8719 ± 0.2339 |
| Recall    | 0.6093 ± 0.2504 |

The results indicate that the baseline U-Net provides reasonable segmentation performance, while performance varies across individual ultrasound images.

## Project Structure

```text
BUSI-Breast-Ultrasound-Segmentation/
│
├── 01_visualize_dataset.py
├── 02_dataset.py
├── 03_split_dataset.py
│
├── dataloader.py
├── loss.py
├── unet.py
│
├── 07_train.py
├── 08_evaluate.py
├── 09_detailed_evaluation.py
│
├── .gitignore
└── README.md
```

### Main Components

**`01_visualize_dataset.py`**

Initial dataset inspection and visualization.

**`02_dataset.py`**

Dataset loading and preprocessing.

**`03_split_dataset.py`**

Creation of training, validation, and test splits.

**`dataloader.py`**

PyTorch DataLoader and batch preparation.

**`unet.py`**

U-Net segmentation architecture.

**`loss.py`**

Loss function implementation.

**`07_train.py`**

Model training and validation.

**`08_evaluate.py`**

Evaluation on the held-out test set.

**`09_detailed_evaluation.py`**

Per-sample metrics and detailed test-set analysis.

## Installation

Python 3.11 was used for development.

Install the required dependencies:

```bash
pip install torch torchvision numpy pillow matplotlib
```

For a CUDA-enabled PyTorch installation, install the appropriate PyTorch version for your NVIDIA GPU and CUDA environment.

## Usage

### 1. Prepare the Dataset

Place the BUSI dataset in the project directory using:

```text
images/
labels/
```

The corresponding image and label files should have matching filenames.

### 2. Inspect the Dataset

```bash
python 01_visualize_dataset.py
```

### 3. Verify Dataset Loading

```bash
python 02_dataset.py
```

### 4. Create Dataset Splits

```bash
python 03_split_dataset.py
```

### 5. Train the Model

```bash
python 07_train.py
```

The best validation checkpoint is saved under:

```text
checkpoints/best_unet.pth
```

### 6. Evaluate the Model

```bash
python 08_evaluate.py
```

### 7. Perform Detailed Evaluation

```bash
python 09_detailed_evaluation.py
```

## Evaluation Metrics

The project evaluates segmentation performance using:

* **Dice Similarity Coefficient (Dice)**
* **Intersection over Union (IoU)**
* **Precision**
* **Recall**

These metrics provide complementary information about the overlap and quality of predicted lesion masks compared with the ground-truth masks.

## Example Prediction

The evaluation pipeline generates visual comparisons between:

```text
Original Ultrasound Image
          ↓
Ground-Truth Segmentation
          ↓
U-Net Prediction
```

Generated evaluation outputs are excluded from the Git repository through `.gitignore`.

## Reproducibility

The project separates the dataset into training, validation, and held-out test sets.

The validation set is used for selecting the best model checkpoint, while the test set is reserved for final evaluation.

## Future Improvements

Potential extensions of this baseline include:

* Data augmentation
* Improved loss functions for class/region imbalance
* Attention-based U-Net variants
* More extensive hyperparameter tuning
* Cross-validation
* Comparison with additional segmentation architectures

## Technologies

* Python
* PyTorch
* NumPy
* PIL
* Matplotlib
* CUDA
* U-Net
* Medical Image Segmentation

## Author

**Mohammadreza Madani**

Biomedical Engineering / Medical AI & Deep Learning

---

This repository is intended as a practical deep learning project demonstrating an end-to-end medical image segmentation workflow using PyTorch and U-Net.
