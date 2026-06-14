import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import DenoisingAutoencoder


# 1. Device setup
# NVIDIA GPU / apple GPU / CPU
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")



# 2. Hyperparameters
# model sees 16 images at once
batch_size = 16
# model goes through the training dataset 30 times
epochs = 30
learning_rate = 0.001
# strength of gaussian noise
noise_factor = 0.08



# 3. Dataset and DataLoader
# converts images to tensors - scale pixels in 0-1 range
transform = transforms.ToTensor()

# loads STL10 training data
train_data = datasets.STL10(
    root="data",
    split="train",
    download=True,
    transform=transform
)

# loads STL10 test data
test_data = datasets.STL10(
    root="data",
    split="test",
    download=True,
    transform=transform
)

# loader gives the batches randomly shuffled
train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

# order doesn't matter for validation, shuffle=False
test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)



# 4. Noise function
# dataset doesn't contain noisy imagaes, we create them manually during training
def add_gaussian_noise(images, noise_factor=0.08):
    noise = noise_factor * torch.randn_like(images)
    noisy_images = images + noise
    noisy_images = torch.clamp(noisy_images, 0.0, 1.0)
    return noisy_images



# 5. Model, loss, optimizer

# creates the model
model = DenoisingAutoencoder().to(device)

# compares output image and clean image pixel by pixel
# we use mean squared error loss because it's reconstruction instead of classification
criterion = nn.MSELoss()
#adam updates the model weights
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)



# 6. Training loop
train_losses = []
val_losses = []

# repeats training for 30 epochs
for epoch in range(epochs):
    model.train()
    #This resets the total training loss at the start of each epoch
    running_train_loss = 0.0

    # each loop gives one batch, ignore labels for denoising
    for images, labels in train_loader:
        clean_images = images.to(device)

        # Add noise
        noisy_images = add_gaussian_noise(clean_images, noise_factor)

        # Forward pass
        outputs = model(noisy_images)

        # Compares denoised output with original clean image
        loss = criterion(outputs, clean_images)

        # Backpropagation, clears old gradients
        optimizer.zero_grad()
        #calculates how each weight contributed to the error
        loss.backward()
        #updates the weights to reduce the error in future
        optimizer.step()

        # adds the current batch’s loss to the total loss for the epoch
        running_train_loss += loss.item()
    #calculates the average for full epoch and stores it    
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    
    # Validation loop
    
    model.eval()
    running_val_loss = 0.0
    #disable gradient calculation   
    with torch.no_grad():
        for images, labels in test_loader:
            clean_images = images.to(device)
            noisy_images = add_gaussian_noise(clean_images, noise_factor)

            outputs = model(noisy_images)
            loss = criterion(outputs, clean_images)

            running_val_loss += loss.item()
    #stores validation loss per epoch 
    avg_val_loss = running_val_loss / len(test_loader)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {avg_train_loss:.6f} "
        f"Val Loss: {avg_val_loss:.6f}"
    )



# 7. Save model
# this saves the model parameters - the learned weights 

torch.save(model.state_dict(), "STL10_denoising_autoencoder.pth")
print("Model saved as STL10_denoising_autoencoder.pth")



# 8. Plot loss
# shows how the model learned over time

plt.figure(figsize=(8, 5))

plt.plot(range(1, epochs + 1), train_losses, label="Training Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss on STL10")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("stl10_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()


#Lates training: Epoch [30/30] Train Loss: 0.001854 Val Loss: 0.001823