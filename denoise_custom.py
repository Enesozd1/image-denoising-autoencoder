# This file turns the model into small usable program in which it
# takes a custom path from the user, resizes the image to 96x96, denoises it,
# saves the output, and saves a comparison image

# denoise_custom.py works best with images that are colored and object-like, similiar
# to what it was trained on - STL10

import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model import DenoisingAutoencoder


# 1. Device setup (Use GPU if available)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


# 2. Settings
model_path = "STL10_denoising_autoencoder.pth"
# create an output folder if it doesn't exist, exist_ok means don't crash if the folder still exists
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)


# 3. Load trained model
model = DenoisingAutoencoder().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# Each custom image must be resized to 96x96 because it was trained on 96x96
# Totensor() changes it to [3,96,96]
transform = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor()
])
# convert model's output tensor back into a normal image
to_pil = transforms.ToPILImage()


# 5. Denoise function
def denoise_image(image_path):
    # Load the custom image, 'RGB' prevents errors from grayscale images
    image = Image.open(image_path).convert("RGB")

    # Keep original filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Convert image to tensor [3,96,96]
    input_tensor = transform(image)

    # Model expects batch dimension, add batch dimension: [3, 96, 96] -> [1, 3, 96, 96]
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # Denoise
    # Run the model without training it
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Remove batch dimension: [1, 3, 96, 96] -> [3, 96, 96]
    output_tensor = output_tensor.squeeze(0).cpu()

    # Keep values between 0 and 1
    output_tensor = torch.clamp(output_tensor, 0, 1)

    # Convert tensor back to image
    denoised_image = to_pil(output_tensor)

    # Save denoised image into 'outputs/image_deniosed.png'
    output_path = os.path.join(output_folder, f"{base_name}_denoised.png")
    denoised_image.save(output_path)

    print(f"Denoised image saved to: {output_path}")

    #return original image, denoised image, and saved file path
    return image, denoised_image, output_path


# 6. Save comparison image (to create side by side image)
def save_comparison(original_image, denoised_image, image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Resize original for fair comparison
    original_resized = original_image.resize((96, 96))

    comparison_path = os.path.join(output_folder, f"{base_name}_comparison.png")

    plt.figure(figsize=(6, 3))
    #shows the input image
    plt.subplot(1, 2, 1)
    plt.imshow(original_resized)
    plt.title("Input Noisy Image")
    plt.axis("off")
    #shows the model output
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image)
    plt.title("Denoised Output")
    plt.axis("off")

    plt.tight_layout()
    #save comparison
    plt.savefig(comparison_path, dpi=200)
    plt.show()

    print(f"Comparison image saved to: {comparison_path}")


# 7. User input
image_path = input("Enter the path of the image you want to denoise: ").strip()

if not os.path.exists(image_path):
    print("Error: Image path does not exist.")
else:
    original, denoised, saved_path = denoise_image(image_path)
    save_comparison(original, denoised, image_path)