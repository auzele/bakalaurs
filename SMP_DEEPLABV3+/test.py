# Model architecture and training loop inspired by:
# Iakubovskii, P. (2019). Segmentation Models Pytorch. GitHub repository.
# https://github.com/qubvel/segmentation_models.pytorch

import os
import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "deeplabv3+_resnet50.pth"
TEST_IMAGES_DIR = ".../test/images"
OUTPUT_DIR = ".../result"


transform = transforms.Compose([
    transforms.ToTensor(),  # only converts to tensor 
])


model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation=None
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()


os.makedirs(OUTPUT_DIR, exist_ok=True)
test_images = os.listdir(TEST_IMAGES_DIR)

for img_name in test_images:
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output)
        pred_mask = (pred_mask > 0.5).float()

   
    pred_mask = pred_mask.squeeze().cpu().numpy() * 255
    pred_mask = Image.fromarray(pred_mask.astype("uint8"))


    save_path = os.path.join(OUTPUT_DIR, f"mask_{img_name}")
    pred_mask.save(save_path)

    print(f"Saved: {save_path}")

print("All test masks saved without resizing!")
