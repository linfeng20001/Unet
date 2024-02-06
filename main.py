import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from SecondTry.model import U_Net

from unet import UNet
from carvana_dataset import CarvanaDataset
from SecondTry.model import U_Net


#youtube: uygar kurt
if __name__ == "__main__":
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 5
    EPOCHS = 2
    DATA_PATH = "C:/Users/Linfe/OneDrive/Desktop/Seg/Dataset"
    MODEL_SAVE_PATH = r"C:/Users/Linfe/OneDrive/Desktop/Seg/Model/model"



    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_dataset = CarvanaDataset(DATA_PATH)





    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    unet = U_Net(3, 1)

    #model = UNet(in_channels=3, num_classes=1).to(device)
    model = U_Net(3, 1)
    optimizer = optim.AdamW(unet.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(EPOCHS)):
        unet.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            y_pred = unet(img)
            optimizer.zero_grad()

            loss = criterion(y_pred, mask)
            train_running_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        train_loss = train_running_loss / (idx + 1)

        unet.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)
                
                y_pred = model(img)
                loss = criterion(y_pred, mask)

                val_running_loss += loss.item()

            val_loss = val_running_loss / (idx + 1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
