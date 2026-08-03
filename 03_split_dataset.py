from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split


# =========================================================
# Configuration
# =========================================================

IMAGE_DIR = Path("images")

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# =========================================================
# Collect image files
# =========================================================

image_files = sorted(IMAGE_DIR.glob("*.png"))

print("Total images:", len(image_files))


# =========================================================
# Create class labels
# =========================================================

# benign_1.png      -> benign
# malignant_1.png   -> malignant

class_labels = [
    image_path.name.split("_")[0]
    for image_path in image_files
]


# =========================================================
# First split: Train + Temporary
# =========================================================

train_files, temp_files, train_labels, temp_labels = train_test_split(
    image_files,
    class_labels,
    test_size=(VAL_RATIO + TEST_RATIO),
    stratify=class_labels,
    random_state=RANDOM_SEED
)


# =========================================================
# Second split: Validation + Test
# =========================================================

relative_test_size = TEST_RATIO / (VAL_RATIO + TEST_RATIO)

val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files,
    temp_labels,
    test_size=relative_test_size,
    stratify=temp_labels,
    random_state=RANDOM_SEED
)


# =========================================================
# Print dataset sizes
# =========================================================

print("\nDataset split:")
print("-------------------------")

print("Train:", len(train_files))
print("Validation:", len(val_files))
print("Test:", len(test_files))
print("Total:", len(train_files) + len(val_files) + len(test_files))


# =========================================================
# Check class distribution
# =========================================================

print("\nClass distribution:")
print("-------------------------")

print("Train:", Counter(train_labels))
print("Validation:", Counter(val_labels))
print("Test:", Counter(test_labels))


# =========================================================
# Check corresponding labels
# =========================================================

LABEL_DIR = Path("labels")

missing_labels = []

for image_path in image_files:

    label_path = LABEL_DIR / image_path.name

    if not label_path.exists():
        missing_labels.append(image_path.name)


print("\nMissing labels:", len(missing_labels))

if missing_labels:
    print("Missing label examples:")
    print(missing_labels[:10])
else:
    print("All images have corresponding labels.")