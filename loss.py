import torch
import torch.nn as nn


# =========================================================
# Dice Loss
# =========================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        # Convert logits to probabilities
        probabilities = torch.sigmoid(logits)

        # Flatten
        probabilities = probabilities.view(-1)
        targets = targets.view(-1)

        # Intersection
        intersection = (probabilities * targets).sum()

        # Dice coefficient
        dice = (
            (2.0 * intersection + self.smooth)
            /
            (
                probabilities.sum()
                + targets.sum()
                + self.smooth
            )
        )

        # Convert Dice to Dice Loss
        return 1.0 - dice


# =========================================================
# Combined BCE + Dice Loss
# =========================================================

class BCEDiceLoss(nn.Module):

    def __init__(self, bce_weight=0.5, dice_weight=0.5):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):

        bce_loss = self.bce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        total_loss = (
            self.bce_weight * bce_loss
            +
            self.dice_weight * dice_loss
        )

        return total_loss


# =========================================================
# Test Loss Functions
# =========================================================

if __name__ == "__main__":

    # Fake model output
    logits = torch.randn(
        8,
        1,
        256,
        256
    )

    # Fake ground-truth masks
    targets = torch.randint(
        0,
        2,
        (
            8,
            1,
            256,
            256
        )
    ).float()

    # Create loss function
    criterion = BCEDiceLoss()

    # Calculate loss
    loss = criterion(
        logits,
        targets
    )

    print("Logits shape:", logits.shape)
    print("Targets shape:", targets.shape)
    print("Loss:", loss.item())