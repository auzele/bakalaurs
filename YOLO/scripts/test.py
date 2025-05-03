import os
import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = ".../weights/best.pt"
IMAGE_DIR = ".../test/images"
OUTPUT_MASKS_DIR = ".../results/binary_masks"


os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)


print("Loading YOLO model...")
model = YOLO(MODEL_PATH)


print("Running YOLO segmentation...")
results = model.predict(
    source=IMAGE_DIR,
    save=False,
    imgsz=640,
    show_boxes=False
)

#BINARY MASKS
print("💾 Saving binary masks...")
for r in results:
    h, w = r.orig_img.shape[:2]
    binary_mask = np.zeros((h, w), dtype=np.uint8)

    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()
        for m in masks:
            mask = (m * 255).astype(np.uint8)
            binary_mask = np.maximum(binary_mask, mask)

    filename = os.path.basename(r.path)
    name = os.path.splitext(filename)[0]
    out_path = os.path.join(OUTPUT_MASKS_DIR, f"{name}.png")

    cv2.imwrite(out_path, binary_mask)

print(f"Binary masks saved to: {OUTPUT_MASKS_DIR}")
