import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

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
        mask = (mask > 0).float()  

        return image, mask
