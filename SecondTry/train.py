import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from PIL import Image
from SecondTry.config import ALL_CLASSES, LABEL_COLORS_LIST, VIS_LABEL_MAP
from SecondTry.dataset import get_images, get_dataset, get_data_loaders
from SecondTry.engine import train, validate
from SecondTry.model import U_Net, prepare_model
import matplotlib.pyplot as plt

LEARNING_RATE = 3e-4
BATCH_SIZE = 5
EPOCHS = 3
img_size = 512
DATA_PATH = "C:/Users/Linfe/OneDrive/Desktop/Seg/Dataset"
MODEL_SAVE_PATH = r"C:/Users/Linfe/OneDrive/Desktop/Seg/Model/model"

device = "cuda" if torch.cuda.is_available() else "cpu"




if __name__ == '__main__':
    model = U_Net(3, len(ALL_CLASSES))
    #model = prepare_model(num_classes=len(ALL_CLASSES))
    #print(model)
    #total_params = sum(p.numel() for p in model.parameters())
    #print(f"{total_params:,} total parameters.")
    #total_trainable_params = sum(
        #p.numel() for p in model.parameters() if p.requires_grad)
    #print(f"{total_trainable_params:,} training parameters.")


    #different optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_images, train_masks, valid_images, valid_masks = get_images(
        root_path=DATA_PATH
    )

    #visualize the training data
    #image = Image.open(train_images[0])
    #mask = Image.open(train_masks[0]).convert("L")  # Convert to grayscale

    #plt.figure(figsize=(10, 5))

    #plt.subplot(1, 2, 1)
    #plt.imshow(image)
    #plt.title('Training Image')

    #plt.subplot(1, 2, 2)
    #plt.imshow(mask, cmap='gray')
    #plt.title('Training Mask')

    #plt.show()
    #print(train_images)

    classes_to_train = ALL_CLASSES
    train_dataset, valid_dataset = get_dataset(
        train_images,
        train_masks,
        valid_images,
        valid_masks,
        ALL_CLASSES,
        classes_to_train,
        LABEL_COLORS_LIST,
        img_size=img_size
    )

    train_dataloader, valid_dataloader = get_data_loaders(
        train_dataset, valid_dataset, batch_size=BATCH_SIZE
    )

    for epoch in range(EPOCHS):
        print(f"EPOCH: {epoch + 1}")
        train_epoch_loss, train_epoch_pixacc, train_epoch_miou = train(
            model,
            train_dataloader,
            device,
            optimizer,
            criterion,
            classes_to_train
        )

        print(
            f"Train Epoch Loss: {train_epoch_loss:.4f},",
            f"Train Epoch PixAcc: {train_epoch_pixacc:.4f},",
            f"Train Epoch mIOU: {train_epoch_miou:4f}"
        )





    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print('TRAINING COMPLETE')