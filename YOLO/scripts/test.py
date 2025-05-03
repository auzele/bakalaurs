import os
import cv2
import numpy as np
from ultralytics import YOLO

# ===== CEĻI =====
MODEL_PATH = "/Users/viktorijaauzele/PycharmProjects/YOLOv8/YOLOv8/scripts/runs/segment/yolov8n_istais/weights/best.pt"
IMAGE_DIR = "/Users/viktorijaauzele/PycharmProjects/YOLOv8/YOLOv8/datasets/test/input"
PROJECT_DIR = "../results"
RUN_NAME = "segmented_batch2"
FILLED_DIR = os.path.join(PROJECT_DIR, RUN_NAME, "filled_masks")

# ===== DIREKTORIJAS =====
os.makedirs(FILLED_DIR, exist_ok=True)

# ===== 1. Ielādē modeli =====
print("\U0001F4E6 Ielādē YOLOv8 segmentācijas modeli...")
model = YOLO(MODEL_PATH)

# ===== 2. Segmentē =====
print("\U0001F9E0 Veic segmentāciju uz visām bildēm mapē...")
results = model.predict(
    source=IMAGE_DIR,
    save=True,
    project=PROJECT_DIR,
    name=RUN_NAME,
    imgsz=640,
    show_boxes=False
)

# ===== 3. Saglabā maskas bez caurspīdīguma =====
print("\U0001F3A8 Zīmē maskas bez caurspīdīguma...")

for i, r in enumerate(results):
    im0 = r.orig_img.copy()
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy()
        for mask in masks:
            mask = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im0, contours, -1, color=(0, 0, 255), thickness=-1)  # aizpilda ar sarkanu

        print(f"Image {i} - {len(masks)} maskas atrastas")
    else:
        print(f"Image {i} - maskas NAV atrastas")

    # Saglabā aizpildīto bildi
    filename = os.path.basename(r.path)
    out_path = os.path.join(FILLED_DIR, f"filled_{filename}")
    cv2.imwrite(out_path, im0)

print(f"\n✅ Saglabāti aizpildītie masku rezultāti: {FILLED_DIR}")
