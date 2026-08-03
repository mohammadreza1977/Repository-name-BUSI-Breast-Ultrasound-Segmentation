from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
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


# =========================================================
# BUSI Dataset
# =========================================================

class BUSIDataset(Dataset):

    def __init__(self, image_dir, label_dir):

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.image_files = sorted(
            self.image_dir.glob("*.png")
        )

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

        # -----------------------------------------
        # Image path
        # -----------------------------------------

        image_path = self.image_files[index]

        # -----------------------------------------
        # Corresponding label
        # -----------------------------------------

        label_path = self.label_dir / image_path.name

        # -----------------------------------------
        # Load image and mask
        # -----------------------------------------

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        # -----------------------------------------
        # Resize image
        # -----------------------------------------

        image = self.image_transform(image)

        # -----------------------------------------
        # Resize mask
        # -----------------------------------------

        label = TF.resize(
            label,
            IMAGE_SIZE,
            interpolation=T.InterpolationMode.NEAREST
        )

        # -----------------------------------------
        # Convert mask to Tensor
        # -----------------------------------------

        label = torch.from_numpy(
            __import__("numpy").array(label)
        ).float()

        # Add channel dimension
        label = label.unsqueeze(0)

        return image, label


# =========================================================
# Test Dataset
# =========================================================

if __name__ == "__main__":

    dataset = BUSIDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR
    )

    print("Number of samples:", len(dataset))

    # Get one sample
    image, label = dataset[0]

    print("\nSingle sample:")
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)

    print("Label shape:", label.shape)
    print("Label dtype:", label.dtype)

    print("Label unique values:", torch.unique(label))


    # =====================================================
    # DataLoader
    # =====================================================

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # Get one batch
    images, labels = next(iter(dataloader))

    print("\nBatch:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)