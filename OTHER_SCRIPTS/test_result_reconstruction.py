import os
from PIL import Image
import numpy as np
import math


INPUT_DIR = "/Users/viktorijaauzele/PycharmProjects/256x256/640"
ORIGINAL_IMAGE = "/Users/viktorijaauzele/PycharmProjects/256x256/1000x1000_ogre.tif"
OUTPUT_IMAGE = "/Users/viktorijaauzele/PycharmProjects/256x256/reconstructed.png"
PATCH_SIZE = 640
OVERLAP = 0.2
STRIDE = int(PATCH_SIZE * (1 - OVERLAP))  # e.g. 512 if OVERLAP = 20%


with Image.open(ORIGINAL_IMAGE) as img:
    orig_width, orig_height = img.size
    mode = img.mode

print(f"Reconstructing image: {orig_width} x {orig_height}, mode: {mode}")


reconstructed = np.zeros((orig_height, orig_width, 3), dtype=np.float32)
weight = np.zeros((orig_height, orig_width, 1), dtype=np.float32)


tile_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".png") and f[-8:-4].isdigit()]
tile_files.sort(key=lambda f: int(f[-8:-4]))

x_tiles = math.ceil((orig_width - PATCH_SIZE) / STRIDE) + 1
y_tiles = math.ceil((orig_height - PATCH_SIZE) / STRIDE) + 1


for idx, filename in enumerate(tile_files):
    row = idx // x_tiles
    col = idx % x_tiles

    x = col * STRIDE
    y = row * STRIDE

    tile_path = os.path.join(INPUT_DIR, filename)
    tile = Image.open(tile_path).convert("RGB")
    tile_np = np.array(tile, dtype=np.float32)

    h, w = tile_np.shape[:2]
    end_y = min(y + h, orig_height)
    end_x = min(x + w, orig_width)

    if end_y <= y or end_x <= x:
        continue

    reconstructed[y:end_y, x:end_x] += tile_np[:end_y - y, :end_x - x]
    weight[y:end_y, x:end_x] += 1


weight[weight == 0] = 1
reconstructed = (reconstructed / weight).astype(np.uint8)


Image.fromarray(reconstructed).save(OUTPUT_IMAGE)
print(f"Reconstructed RGB image saved to: {OUTPUT_IMAGE}")