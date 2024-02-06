import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from carvana_dataset import CarvanaDataset
import torch
from unet import UNet
import numpy as np
from SecondTry.model import U_Net
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
# Replace this with the path to your dataset
DATA_PATH = "C:/Users/Linfe/OneDrive/Desktop/Seg/Dataset"

MODEL_PATH = "C:/Users/Linfe/OneDrive/Desktop/Seg/Model/model"
train_dataset = CarvanaDataset(DATA_PATH)

model = U_Net(3, 1).to(device)
model.load_state_dict(torch.load(MODEL_PATH))

model.eval()

train_dataset = CarvanaDataset(DATA_PATH)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=3,
                              shuffle=True)

for idx, img_mask in enumerate(tqdm(train_dataloader)):
    img = img_mask[0].float().to(device)
    mask = img_mask[1].float().to(device)
    pr_mask = model(img)

    img_np = img[idx].permute(1, 2, 0).numpy()
    mask_np = mask[idx].squeeze(0).numpy()

    pr_mask_np = pr_mask[idx].detach().numpy()
    #to_pil = transforms.ToPILImage()
    #pr_mask_np = to_pil(pr_mask)


    print(pr_mask_np.dtype)
    print(pr_mask_np.shape)
    print(np.unique(pr_mask_np, return_counts=True))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[1].imshow(mask_np, cmap="gray")
    #axes[2].imshow(pr_mask_np, cmap="gray")
    plt.show()
    exit()

