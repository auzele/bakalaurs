# Model architecture and training loop inspired by:
# Iakubovskii, P. (2019). Segmentation Models Pytorch. GitHub repository.
# https://github.com/qubvel/segmentation_models.pytorch



from dataset import BuildingDataset
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

train_dataset = BuildingDataset("train/images", "train/masks")
val_dataset = BuildingDataset("validation/images", "validation/masks")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = smp.DeepLabV3Plus(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)

loss_fn = smp.losses.DiceLoss(mode="binary")
optimizer = torch.optim.Adam(model.parameters())

best_val_loss = float("inf")

for epoch in range(50):
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

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/50 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_deeplabv3+_resnet50.pth")
        print(">> Best model saved!")
