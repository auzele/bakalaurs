import os
import torch
from PIL import Image
import segmentation_models_pytorch as smp
import numpy as np

#PARAMETERS
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "unet_resnet50.pth"  
TEST_IMAGES_DIR = ".../test/images"  
OUTPUT_DIR = ".../test/result"  

#MODEL
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

#BUILDING PREDICTION
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_images = os.listdir(TEST_IMAGES_DIR)

for img_name in test_images:
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    image = Image.open(img_path).convert("RGB")

    input_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)  

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output)
        pred_mask = (pred_mask > 0.5).float()

    pred_mask = pred_mask.squeeze().cpu().numpy() * 255
    pred_mask = Image.fromarray(pred_mask.astype("uint8"))
    pred_mask = pred_mask.resize(image.size)

    save_path = os.path.join(OUTPUT_DIR, f"mask_{img_name}")
    pred_mask.save(save_path)

    print(f"Saved: {save_path}")

print("\n Finished!")
