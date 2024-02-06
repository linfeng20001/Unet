import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from carvana_dataset import CarvanaDataset
from unet import UNet
from SecondTry.model import U_Net


def pred_show_image_grid(data_path, model_pth, device):
    model = U_Net(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))
    image_dataset = CarvanaDataset(data_path, test=True)
    images = []
    orig_masks = []
    pred_masks = []

    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        pred_mask = model(img)

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1, 2, 0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1

        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1, 2, 0)

        images.append(img)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)

    images.extend(orig_masks)
    images.extend(pred_masks)
    fig = plt.figure()
    for i in range(1, 3 * len(image_dataset) + 1):
        fig.add_subplot(3, len(image_dataset), i)
        plt.imshow(images[i - 1], cmap="gray")
    plt.show()


def single_image_inference(image_pth, model_pth,mask_pth, device):
    model = U_Net(3, 1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))


    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    img = transform(Image.open(image_pth)).float().to(device)
    img = img.unsqueeze(0)
    mask = transform(Image.open(mask_pth)).float().to(device)
    mask = mask.unsqueeze(0)

    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    mask = mask.squeeze(0).cpu().detach()
    mask = mask.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    #pred_mask[pred_mask < 0] = 0
    #pred_mask[pred_mask > 0] = 1

    fig = plt.figure()

    for i in range(1, 4):  # Ändere die Range auf 4 für drei Bilder
        fig.add_subplot(1, 3, i)  # Ändere die Anzahl der Subplots auf 3
        if i == 1:
            plt.imshow(img, cmap="gray")
            plt.title("Original Image")
        elif i == 2:
            plt.imshow(pred_mask, cmap="gray")
            plt.title("Predicted Mask")
        else:
            plt.imshow(mask, cmap="gray")
            plt.title("Original Mask")

    plt.show()


if __name__ == "__main__":
    #"C:\Users\Linfe\OneDrive\Desktop\Seg\Dataset\train\hamburg_000000_000042_leftImg8bit.png"
    #"C:\Users\Linfe\OneDrive\Desktop\Seg\Dataset\train_masks\hamburg_000000_106102_gtFine_labelIds.png"
    #"C:\Users\Linfe\OneDrive\Desktop\Seg\Model\model"
    SINGLE_IMG_PATH = "C:/Users/Linfe/OneDrive/Desktop/Seg/Dataset/images/train/hamburg_000000_007737_leftImg8bit.png"
    MASK_PATH = "C:/Users/Linfe/OneDrive/Desktop/Seg/Dataset/labels/train/hamburg_000000_007737_gtFine_color.png"
    #DATA_PATH = "C:/Users/Linfe/OneDrive/Desktop/Seg/Dataset"
    MODEL_PATH = "C:/Users/Linfe/OneDrive/Desktop/Seg/Model/model"
    SAVE_PATH = "C:/Users/Linfe/OneDrive/Desktop/Seg/result"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, MASK_PATH, device)
