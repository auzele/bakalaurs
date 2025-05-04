import os
from PIL import Image
import math


INPUT_IMAGE = ".../4311-15_1.tif"
OUTPUT_DIR = ".../640"
PATCH_SIZE = 640


# Use PATCH_SIZE for no overlap. For overlap, use:
# OVERLAP = 0.1
# STRIDE = int(PATCH_SIZE * (1 - OVERLAP))
STRIDE = PATCH_SIZE  # No overlap


os.makedirs(OUTPUT_DIR, exist_ok=True)


image = Image.open(INPUT_IMAGE)
mode = image.mode

width, height = image.size
print(f" Input image size: {width} x {height}, mode: {mode}")


x_tiles = math.ceil(width / STRIDE)
y_tiles = math.ceil(height / STRIDE)

count = 0
for y in range(y_tiles):
    for x in range(x_tiles):
        left = x * STRIDE
        upper = y * STRIDE
        right = left + PATCH_SIZE
        lower = upper + PATCH_SIZE

        # Crop the tile, clipping if near the edge
        tile = image.crop((left, upper, min(right, width), min(lower, height)))

        # Pad with black if tile is smaller than PATCH_SIZE
        padded_tile = Image.new(mode, (PATCH_SIZE, PATCH_SIZE))
        padded_tile.paste(tile, (0, 0))

        # Save tile with indexed filename
        tile_name = f"4311-15_1_{count:04d}.png"
        padded_tile.save(os.path.join(OUTPUT_DIR, tile_name))
        count += 1

print(f"Saved {count} tiles ({PATCH_SIZE}x{PATCH_SIZE}) in original mode → {OUTPUT_DIR}")
