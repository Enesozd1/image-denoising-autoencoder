import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. Transform images into tensors
#uses [channels, height, width] STL10 uses [3,96,96]
transform = transforms.ToTensor()

# 2. Load STL10 training dataset
#dataset will be stored in folder 'data'
train_data = datasets.STL10(
    root="data",
    split="train",
    download=True,
    transform=transform
)
#transform=transform indicates that each image will be converted into tensor

# 3. Create DataLoader
#creates batches of 4 images, so when you take one batch the image should be [4,3,96,96]
#Meaning 4 images, 3 color channels, 96 height and width
train_loader = DataLoader(
    train_data,
    batch_size=4,
    shuffle=True
)
#shuffle=True means the images are randomly mixed each time


# 4. Function to add Gaussian noise
# Creates random noise with the exact same shape as the image batch
def add_gaussian_noise(images, noise_factor=0.08):
    noise = noise_factor * torch.randn_like(images)
    #Adds noise to the clean original images
    noisy_images = images + noise
    #Keeps pixel values in the range of [0.0,1.0]
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
    return noisy_images


# 5. Function to display images
def show_images(clean_images, noisy_images):
    #Move tensors to CPU, matplotlib can't display CUDA tensors
    clean_images = clean_images.cpu()
    noisy_images = noisy_images.cpu()

    plt.figure(figsize=(12, 4))

    for i in range(8):
        # clean image
        plt.subplot(2, 8, i + 1)
        #Change pytorch image shape [channels,height,width] to matplotlib's expectation: [height,width,channel]
        img = clean_images[i].permute(1, 2, 0)
        plt.imshow(img, interpolation="nearest")
        plt.title("Clean")
        plt.axis("off")

        # noisy image
        plt.subplot(2, 8, i + 9)
        noisy_img = noisy_images[i].permute(1, 2, 0)
        plt.imshow(noisy_img, interpolation="nearest")
        plt.title("Noisy")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# 6. Get one batch
#images contains the images, we don't need STL10 class labels for denoising
images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

# 7. Add noise
noisy_images = add_gaussian_noise(images, noise_factor=0.08)

# 8. Show clean and noisy images
show_images(images, noisy_images)