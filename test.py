import os
import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp

# =================== PARAMETRI ===================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "deeplabv3+_resnet50.pth"  # Saglabātā modeļa fails
TEST_IMAGES_DIR = "/Users/viktorijaauzele/PycharmProjects/SMP_DEEPLABV3+/test/input"    # Mape ar test attēliem
OUTPUT_DIR = "/Users/viktorijaauzele/PycharmProjects/SMP_DEEPLABV3+/test/result"  # Kur saglabāt maskas

# =================== TRANSFORMĀCIJAS ===================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# =================== MODELA IELĀDE ===================
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

# =================== TEST ATTĒLU PROGNOZĒŠANA ===================
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_images = os.listdir(TEST_IMAGES_DIR)

for img_name in test_images:
    img_path = os.path.join(TEST_IMAGES_DIR, img_name)
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output)
        pred_mask = (pred_mask > 0.5).float()

    pred_mask = pred_mask.squeeze().cpu().numpy() * 255
    pred_mask = Image.fromarray(pred_mask.astype("uint8"))
    pred_mask = pred_mask.resize(image.size)

    save_path = os.path.join(OUTPUT_DIR, f"mask_{img_name}")
    pred_mask.save(save_path)

    print(f"Saglabāts: {save_path}")

print("\nTests pabeigts!")
