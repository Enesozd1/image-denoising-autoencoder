# This file confirms tensor shapes are valid, to make sure MSELoss doesn't  fail
import torch
from model import DenoisingAutoencoder

# create the model for testing the shape
model = DenoisingAutoencoder()

# 'fake' image data 4 images, 3 channels, 96 height and width
dummy_input = torch.randn(4, 3, 96, 96)

# sends fake image through autoencoder
# output is meaningless visually (untrained model), we care about the shape
output = model(dummy_input)

# Expected output: 
# Input shape: torch.Size([4, 3, 96, 96])
# Output shape: torch.Size([4, 3, 96, 96])

print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)

