import os
import cv2


MASK_DIR = ".../640_mask"
OUTPUT_TXT_DIR = ".../yolo_labels"
IMAGE_SIZE = 640

CLASS_ID = 0


os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)


mask_files = [f for f in os.listdir(MASK_DIR) if f.endswith(".png")]
mask_files.sort()

for filename in mask_files:
    path = os.path.join(MASK_DIR, filename)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Skipping unreadable file: {filename}")
        continue


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    txt_lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)


        if w < 3 or h < 3:
            continue


        x_center = (x + w / 2) / IMAGE_SIZE
        y_center = (y + h / 2) / IMAGE_SIZE
        norm_w = w / IMAGE_SIZE
        norm_h = h / IMAGE_SIZE

        txt_line = f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
        txt_lines.append(txt_line)


    txt_filename = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(OUTPUT_TXT_DIR, txt_filename)
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))

    print(f" Created YOLO label: {txt_filename} ({len(txt_lines)} objects)")
