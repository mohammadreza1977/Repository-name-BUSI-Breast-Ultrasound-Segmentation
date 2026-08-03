import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from dataloader import test_loader
from unet import UNet


# =========================================================
# Configuration
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = Path("checkpoints/best_unet.pth")
OUTPUT_DIR = Path("evaluation_results")

OUTPUT_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.5


# =========================================================
# Metrics
# =========================================================

def calculate_metrics(predictions, targets, threshold=0.5):

    probabilities = torch.sigmoid(predictions)

    predicted_masks = (
        probabilities >= threshold
    ).float()

    targets = targets.float()

    # Flatten
    predicted_masks = predicted_masks.view(-1)
    targets = targets.view(-1)

    # Confusion components
    true_positive = (
        predicted_masks * targets
    ).sum()

    false_positive = (
        predicted_masks * (1 - targets)
    ).sum()

    false_negative = (
        (1 - predicted_masks) * targets
    ).sum()

    true_negative = (
        (1 - predicted_masks) * (1 - targets)
    ).sum()

    smooth = 1e-6

    # Dice
    dice = (
        2 * true_positive + smooth
    ) / (
        2 * true_positive
        + false_positive
        + false_negative
        + smooth
    )

    # IoU
    iou = (
        true_positive + smooth
    ) / (
        true_positive
        + false_positive
        + false_negative
        + smooth
    )

    # Precision
    precision = (
        true_positive + smooth
    ) / (
        true_positive
        + false_positive
        + smooth
    )

    # Recall
    recall = (
        true_positive + smooth
    ) / (
        true_positive
        + false_negative
        + smooth
    )

    return (
        dice.item(),
        iou.item(),
        precision.item(),
        recall.item()
    )


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
        map_location=DEVICE
    )
)

model.eval()

print("Device:", DEVICE)
print("Model loaded:", MODEL_PATH)


# =========================================================
# Test Evaluation
# =========================================================

all_dice = []
all_iou = []
all_precision = []
all_recall = []

saved_visualizations = 0
MAX_VISUALIZATIONS = 5


with torch.no_grad():

    for images, masks in test_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Model prediction
        predictions = model(images)

        # Metrics
        dice, iou, precision, recall = calculate_metrics(
            predictions,
            masks,
            threshold=THRESHOLD
        )

        all_dice.append(dice)
        all_iou.append(iou)
        all_precision.append(precision)
        all_recall.append(recall)

        # =================================================
        # Visualization
        # =================================================

        probabilities = torch.sigmoid(predictions)

        predicted_masks = (
            probabilities >= THRESHOLD
        ).float()

        for i in range(images.shape[0]):

            if saved_visualizations >= MAX_VISUALIZATIONS:
                break

            image = images[i].cpu().permute(
                1, 2, 0
            ).numpy()

            ground_truth = masks[i].cpu().squeeze().numpy()

            prediction = predicted_masks[i].cpu().squeeze().numpy()

            plt.figure(figsize=(12, 4))

            # Original image
            plt.subplot(1, 3, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            # Ground truth
            plt.subplot(1, 3, 2)
            plt.imshow(ground_truth, cmap="gray")
            plt.title("Ground Truth")
            plt.axis("off")

            # Prediction
            plt.subplot(1, 3, 3)
            plt.imshow(prediction, cmap="gray")
            plt.title("Prediction")
            plt.axis("off")

            plt.tight_layout()

            save_path = (
                OUTPUT_DIR
                / f"prediction_{saved_visualizations + 1}.png"
            )

            plt.savefig(
                save_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close()

            saved_visualizations += 1


# =========================================================
# Final Results
# =========================================================

mean_dice = np.mean(all_dice)
mean_iou = np.mean(all_iou)
mean_precision = np.mean(all_precision)
mean_recall = np.mean(all_recall)


print("\n========================================")
print("Test Results")
print("========================================")

print(f"Dice Score : {mean_dice:.4f}")
print(f"IoU Score  : {mean_iou:.4f}")
print(f"Precision  : {mean_precision:.4f}")
print(f"Recall     : {mean_recall:.4f}")

print("\nVisualizations saved to:")
print(OUTPUT_DIR)