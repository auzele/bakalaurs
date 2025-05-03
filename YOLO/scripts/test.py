import os
import cv2
import numpy as np
from ultralytics import YOLO


MODEL_PATH = ".../weights/best.pt"
IMAGE_DIR = ".../test/images"
PROJECT_DIR = ".../test/results"
RUN_NAME = "segmented_batch2"
MASK_DIR = os.path.join(PROJECT_DIR, RUN_NAME, "binary_masks")


os.makedirs(MASK_DIR, exist_ok=True)

#MODEL
model = YOLO(MODEL_PATH)


results = model.predict(
    source=IMAGE_DIR,
    save=False,
    imgsz=640,
    show_boxes=False
)

#BINARY MASKS

for i, r in enumerate(results):
    h, w = r.orig_img.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)  

    if r.masks is not None:
        masks = r.masks.data.cpu().numpy() 
        for mask in masks:
            binary = (mask * 255).astype(np.uint8) 
            combined_mask = np.maximum(combined_mask, binary)  

        filename = os.path.basename(r.path).rsplit(".", 1)[0]
        mask_path = os.path.join(MASK_DIR, f"{filename}_combined_mask.png")
        cv2.imwrite(mask_path, combined_mask)
        print(f"{len(masks)} masks merged for image: {r.path}")
    else:
        print(f"No masks found for image: {r.path}")

print(f"Merged binary masks saved in: {MASK_DIR}")
