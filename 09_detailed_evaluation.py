import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from dataloader import test_loader
from unet import UNet


# =========================================================
# Configuration
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path("checkpoints/best_unet.pth")

OUTPUT_DIR = Path("detailed_evaluation")
OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.5


# =========================================================
# Load Model
# =========================================================

model = UNet(
    in_channels=3,
    out_channels=1
).to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("Device:", DEVICE)
print("Model loaded:", MODEL_PATH)


# =========================================================
# Metric Function
# =========================================================

def calculate_metrics(prediction, target):

    prediction = (torch.sigmoid(prediction) >= THRESHOLD).float()
    target = target.float()

    prediction = prediction.view(-1)
    target = target.view(-1)

    tp = (prediction * target).sum().item()
    fp = (prediction * (1 - target)).sum().item()
    fn = ((1 - prediction) * target).sum().item()

    smooth = 1e-6

    dice = (
        (2 * tp + smooth)
        /
        (2 * tp + fp + fn + smooth)
    )

    iou = (
        (tp + smooth)
        /
        (tp + fp + fn + smooth)
    )

    precision = (
        (tp + smooth)
        /
        (tp + fp + smooth)
    )

    recall = (
        (tp + smooth)
        /
        (tp + fn + smooth)
    )

    return dice, iou, precision, recall


# =========================================================
# Evaluation
# =========================================================

all_results = []

class_results = defaultdict(list)

sample_index = 0


with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        predictions = model(images)

        for i in range(images.shape[0]):

            dice, iou, precision, recall = calculate_metrics(
                predictions[i:i + 1],
                masks[i:i + 1]
            )

            # Recover image name from dataset
            image_path = test_loader.dataset.image_files[sample_index]

            filename = image_path.name

            # Class from filename
            class_name = filename.split("_")[0]

            result = {
                "filename": filename,
                "class": class_name,
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall
            }

            all_results.append(result)

            class_results[class_name].append(result)

            sample_index += 1


# =========================================================
# Overall Statistics
# =========================================================

def print_statistics(results, title):

    dice = [r["dice"] for r in results]
    iou = [r["iou"] for r in results]
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]

    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)

    print(f"Samples     : {len(results)}")

    print(
        f"Dice        : {np.mean(dice):.4f} "
        f"+/- {np.std(dice):.4f}"
    )

    print(
        f"IoU         : {np.mean(iou):.4f} "
        f"+/- {np.std(iou):.4f}"
    )

    print(
        f"Precision   : {np.mean(precision):.4f} "
        f"+/- {np.std(precision):.4f}"
    )

    print(
        f"Recall      : {np.mean(recall):.4f} "
        f"+/- {np.std(recall):.4f}"
    )


# =========================================================
# Print Overall Results
# =========================================================

print_statistics(
    all_results,
    "Overall Test Results"
)


# =========================================================
# Print Per-Class Results
# =========================================================

for class_name in sorted(class_results.keys()):

    print_statistics(
        class_results[class_name],
        f"Class: {class_name.upper()}"
    )


# =========================================================
# Best / Worst Samples
# =========================================================

sorted_results = sorted(
    all_results,
    key=lambda x: x["dice"]
)

worst_samples = sorted_results[:5]
best_samples = sorted_results[-5:][::-1]


print("\n" + "=" * 55)
print("Worst 5 Samples")
print("=" * 55)

for i, result in enumerate(worst_samples, 1):

    print(
        f"{i}. {result['filename']} | "
        f"Class: {result['class']} | "
        f"Dice: {result['dice']:.4f} | "
        f"IoU: {result['iou']:.4f}"
    )


print("\n" + "=" * 55)
print("Best 5 Samples")
print("=" * 55)

for i, result in enumerate(best_samples, 1):

    print(
        f"{i}. {result['filename']} | "
        f"Class: {result['class']} | "
        f"Dice: {result['dice']:.4f} | "
        f"IoU: {result['iou']:.4f}"
    )


# =========================================================
# Save Results to TXT
# =========================================================

output_file = OUTPUT_DIR / "detailed_results.txt"

with open(output_file, "w", encoding="utf-8") as f:

    f.write("BUSI U-Net Detailed Evaluation\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total Test Samples: {len(all_results)}\n\n")

    for result in all_results:

        f.write(
            f"{result['filename']}\t"
            f"{result['class']}\t"
            f"Dice={result['dice']:.4f}\t"
            f"IoU={result['iou']:.4f}\t"
            f"Precision={result['precision']:.4f}\t"
            f"Recall={result['recall']:.4f}\n"
        )

print("\nDetailed results saved to:")
print(output_file)