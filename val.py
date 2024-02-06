import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from carvana_dataset import CarvanaDataset
from unet import UNet

device = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "C:/Users/Linfe/OneDrive/Desktop/Seg/Dataset"
MODEL_PATH = "C:/Users/Linfe/OneDrive/Desktop/Seg/Model/model"
train_dataset = CarvanaDataset(DATA_PATH)

model = UNet(in_channels=3, num_classes=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device(device)))

# Assuming 'model' is your trained model
model.eval()

# Choose an index from your dataset for visualization
index = 0

# Get the original image and mask from the dataset
original_img, original_mask = train_dataset[index]

# Convert the original image and mask to NumPy arrays for visualization
original_img_np = original_img.permute(1, 2, 0).numpy()
original_mask_np = original_mask.squeeze(0).numpy()

# Forward pass to get the model prediction
# ...

# Forward pass to get the model prediction
#with torch.no_grad():
#    input_tensor = original_img.unsqueeze(0).to(device)
#    predicted_mask = model(input_tensor).cpu().squeeze(0).numpy()

# Threshold the predicted mask (assuming binary segmentation)


#predicted_mask[predicted_mask < 0.5] = 0
#predicted_mask[predicted_mask >= 0.5] = 1



# Display the original image, original mask, and predicted mask side by side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_img_np.squeeze().astype('uint8'))
axes[0].set_title(f"Original Image: {os.path.basename(train_dataset.images[index])}")

axes[1].imshow(original_mask_np, cmap="gray")
axes[1].set_title(f"Original Mask: {os.path.basename(train_dataset.masks[index])}")

#axes[2].imshow(predicted_mask, cmap="gray")
#axes[2].set_title("Predicted Mask")

plt.show()
