import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from dataset import BuildingDataset

#PARAMETERS
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 50


train_loader = DataLoader(BuildingDataset("train/images", "train/masks"), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(BuildingDataset("validation/images", "validation/masks"), batch_size=BATCH_SIZE)

#MODEL
model = smp.Unet(
    encoder_name="resnet50",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)


loss_fn = smp.losses.DiceLoss(mode="binary")
optimizer = torch.optim.Adam(model.parameters())

#TRAIN
best_val_loss = float("inf")
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

    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_unet.pth")
        print("Best model saved!")
