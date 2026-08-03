from PIL import Image
import matplotlib.pyplot as plt

image_path = "images/benign_1.png"
label_path = "labels/benign_1.png"

image = Image.open(image_path)
label = Image.open(label_path)

print("Image size:", image.size)
print("Image mode:", image.mode)

print("Label size:", label.size)
print("Label mode:", label.mode)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Ultrasound Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(label, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.tight_layout()
plt.show()