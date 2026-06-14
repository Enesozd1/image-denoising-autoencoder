#This file loads the saved model and visually tests it on STL10 images
#Load STL10 test data -> add noise -> pass it through the trained model -> display the result

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import DenoisingAutoencoder


# 1. Device setup use GPU if available 
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


# 2. Settings
# 4 columns for visualization plot
batch_size = 4
noise_factor = 0.08
#model path points to the saved trained model
model_path = "STL10_denoising_autoencoder.pth"


# 3. Load STL10 test data
transform = transforms.ToTensor()

test_data = datasets.STL10(
    root="data",
    split="test",
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=True
)


# 4. Noise function
def add_gaussian_noise(images, noise_factor=0.08):
    noise = noise_factor * torch.randn_like(images)
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
    return noisy_images


# 5. Load trained model
model = DenoisingAutoencoder().to(device)
#Load the saved weights from the training
model.load_state_dict(torch.load(model_path, map_location=device))
#using the model for inference not training
model.eval()


# 6. Get one batch and denoise (labels not needed for denoising)
images, labels = next(iter(test_loader))

# 'clean_images' is the original target and 'noisy_images' is the input
clean_images = images.to(device)
noisy_images = add_gaussian_noise(clean_images, noise_factor)

# no gradient needed because we are not training
with torch.no_grad():
    denoised_images = model(noisy_images)


# Move to CPU for plotting (Matplotlib needs CPU tensors)
clean_images = clean_images.cpu()
noisy_images = noisy_images.cpu()
denoised_images = denoised_images.cpu()


# 7. Show results
# creates the window
plt.figure(figsize=(12, 9))

# loop through each image, for each image there are 3 rows: Noisy, Denoised, Clean. 
for i in range(batch_size):
    # Noisy image
    plt.subplot(3, batch_size, i + 1)
    # convert [3,96,96] into [96,96,3] so Matplotlib can display it
    img = noisy_images[i].permute(1, 2, 0)
    plt.imshow(img, interpolation="bicubic")
    plt.title("Noisy")
    plt.axis("off")

    # Denoised image
    plt.subplot(3, batch_size, i + 1 + batch_size)
    img = denoised_images[i].permute(1, 2, 0)
    plt.imshow(img, interpolation="bicubic")
    plt.title("Denoised")
    plt.axis("off")

    # Clean image
    plt.subplot(3, batch_size, i + 1 + batch_size * 2)
    img = clean_images[i].permute(1, 2, 0)
    plt.imshow(img, interpolation="bicubic")
    plt.title("Clean")
    plt.axis("off")

plt.tight_layout()
plt.savefig("stl10_visual_results.png", dpi=300, bbox_inches="tight")
plt.show()