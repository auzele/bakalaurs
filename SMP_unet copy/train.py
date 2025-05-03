import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp

# =================== PARAMETRI ===================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.0001

# Ievadi savus ceļus šeit
TRAIN_IMAGES_DIR = "/Users/viktorijaauzele/PycharmProjects/SMP_unet/train/images"
TRAIN_MASKS_DIR = "/Users/viktorijaauzele/PycharmProjects/SMP_unet/train/masks"
VAL_IMAGES_DIR = "/Users/viktorijaauzele/PycharmProjects/SMP_unet/validation/images"
VAL_MASKS_DIR = "/Users/viktorijaauzele/PycharmProjects/SMP_unet/validation/masks"

# =================== DATASET KLASE ===================
class BuildingDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        mask = (mask > 0).float()  # Binārā maska

        return image, mask

# =================== DATU IELĀDE ===================
train_dataset = BuildingDataset(
    image_dir=TRAIN_IMAGES_DIR,
    mask_dir=TRAIN_MASKS_DIR
)

valid_dataset = BuildingDataset(
    image_dir=VAL_IMAGES_DIR,
    mask_dir=VAL_MASKS_DIR
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =================== MODELA DEFINĪCIJA ===================
model = smp.Unet(
    encoder_name="resnet50",          # ResNet-50 enkoderis
    encoder_weights="imagenet",       # Pretrained uz ImageNet
    in_channels=3,                     # RGB attēli
    classes=1,                         # Binārā segmentācija
    activation=None
)

model = model.to(DEVICE)

# =================== LOSS UN OPTIMIZATORS ===================
loss_fn = smp.losses.DiceLoss(mode="binary")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =================== TRENIŅA UN VALIDĀCIJAS CIKLS ===================
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in valid_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_loader)

    print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# =================== MODEĻA SAGLABĀŠANA ===================
torch.save(model.state_dict(), "unet_resnet50.pth")
print("Modelis saglabāts!")
