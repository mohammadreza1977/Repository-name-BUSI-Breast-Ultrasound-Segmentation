import torch
import torch.optim as optim
from pathlib import Path

from dataloader import train_loader, val_loader
from unet import UNet
from loss import BCEDiceLoss


# =========================================================
# Configuration
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NUM_EPOCHS = 30
LEARNING_RATE = 1e-4

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best_unet.pth"


# =========================================================
# Model
# =========================================================

model = UNet(
    in_channels=3,
    out_channels=1
).to(DEVICE)


# =========================================================
# Loss
# =========================================================

criterion = BCEDiceLoss()


# =========================================================
# Optimizer
# =========================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# Training
# =========================================================

best_val_loss = float("inf")


for epoch in range(NUM_EPOCHS):

    # -----------------------------------------------------
    # Training mode
    # -----------------------------------------------------

    model.train()

    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = criterion(
            predictions,
            masks
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            predictions = model(images)

            loss = criterion(
                predictions,
                masks
            )

            val_loss += loss.item()

    val_loss /= len(val_loader)


    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Val Loss: {val_loss:.4f}"
    )


    # -----------------------------------------------------
    # Save best model
    # -----------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH
        )

        print(
            f"  → Best model saved "
            f"(Val Loss: {val_loss:.4f})"
        )


print("\nTraining finished.")
print("Best model:", BEST_MODEL_PATH)