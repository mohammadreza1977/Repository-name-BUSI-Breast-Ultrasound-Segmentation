from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# =========================================================
# Configuration
# =========================================================

IMAGE_DIR = Path("images")
LABEL_DIR = Path("labels")

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
RANDOM_SEED = 42


# =========================================================
# Reproducibility
# =========================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# Create Train / Validation / Test Split
# =========================================================

image_files = sorted(IMAGE_DIR.glob("*.png"))

class_labels = [
    image_path.name.split("_")[0]
    for image_path in image_files
]

train_files, temp_files, train_labels, temp_labels = train_test_split(
    image_files,
    class_labels,
    test_size=0.30,
    stratify=class_labels,
    random_state=RANDOM_SEED
)

val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files,
    temp_labels,
    test_size=0.50,
    stratify=temp_labels,
    random_state=RANDOM_SEED
)


# =========================================================
# BUSI Dataset
# =========================================================

class BUSIDataset(Dataset):

    def __init__(self, image_files, label_dir):

        self.image_files = image_files
        self.label_dir = Path(label_dir)

        self.image_transform = T.Compose([
            T.Resize(
                IMAGE_SIZE,
                interpolation=T.InterpolationMode.BILINEAR
            ),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):

        image_path = self.image_files[index]

        label_path = self.label_dir / image_path.name

        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        # Image preprocessing
        image = self.image_transform(image)

        # Mask preprocessing
        label = TF.resize(
            label,
            IMAGE_SIZE,
            interpolation=T.InterpolationMode.NEAREST
        )

        label = torch.from_numpy(
            np.array(label)
        ).float()

        # Add channel dimension
        label = label.unsqueeze(0)

        return image, label


# =========================================================
# Create Datasets
# =========================================================

train_dataset = BUSIDataset(
    train_files,
    LABEL_DIR
)

val_dataset = BUSIDataset(
    val_files,
    LABEL_DIR
)

test_dataset = BUSIDataset(
    test_files,
    LABEL_DIR
)


# =========================================================
# Create DataLoaders
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================================================
# Test Pipeline
# =========================================================

if __name__ == "__main__":

    print("Dataset sizes:")
    print("-------------------------")
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    print("\nClass distribution:")
    print("-------------------------")
    print("Train:", Counter(train_labels))
    print("Validation:", Counter(val_labels))
    print("Test:", Counter(test_labels))

    # Get one training batch
    images, masks = next(iter(train_loader))

    print("\nTraining batch:")
    print("-------------------------")
    print("Images shape:", images.shape)
    print("Masks shape:", masks.shape)

    print("\nData types:")
    print("Images:", images.dtype)
    print("Masks:", masks.dtype)

    print("\nMask values:")
    print(torch.unique(masks))