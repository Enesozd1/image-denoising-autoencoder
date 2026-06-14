# STL10 Image Denoising Using a Convolutional Autoencoder

This project implements a color image denoising pipeline using a convolutional autoencoder trained on the STL10 dataset. The model receives a noisy RGB image as input and learns to reconstruct a cleaner version of the same image.

The project was developed for the AI-LAB course: Computer Vision, Signal Processing, and Natural Language Processing.

---

## 1. Project Goal

The goal of this project is to train a neural network that can remove artificial Gaussian noise from color images.

The model learns the following mapping:

```text
Noisy image → Convolutional Autoencoder → Denoised image
```

The clean image is used as the reconstruction target during training.

---

## 2. Dataset

The project uses the STL10 dataset.

Dataset properties:

```text
Image type: RGB color images
Image size: 96 × 96
Tensor shape: [3, 96, 96]
Training split: 5,000 images
Test split: 8,000 images
Number of classes: 10
```

Although STL10 is originally a classification dataset, the class labels are ignored in this project because the task is image reconstruction, not classification.

---

## 3. Noise Generation

The dataset does not provide noisy-clean image pairs. Therefore, noisy images are created manually by adding Gaussian noise to clean images.

The noisy image is generated as:

```text
x_noisy = x_clean + noise_factor × random_noise
```

In this project:

```text
noise_factor = 0.08
```

After noise is added, the pixel values are clipped to stay inside the valid range:

```text
[0, 1]
```

---

## 4. Model Architecture

The model is defined in:

```text
model.py
```

The model is a convolutional denoising autoencoder with two main parts:

```text
Encoder → Bottleneck → Decoder
```

### Encoder

The encoder extracts image features and compresses the spatial representation.

Main layers:

```text
Conv2d 3 → 32 + ReLU
Conv2d 32 → 64 + ReLU
MaxPool2d
Conv2d 64 → 128 + ReLU
MaxPool2d
```

The encoder transforms the input approximately as:

```text
[3, 96, 96] → [128, 24, 24]
```

### Bottleneck

The bottleneck is the compressed feature representation:

```text
[128, 24, 24]
```

### Decoder

The decoder reconstructs the image from the compressed representation.

Main layers:

```text
ConvTranspose2d 128 → 64 + ReLU
Conv2d 64 → 64 + ReLU
ConvTranspose2d 64 → 32 + ReLU
Conv2d 32 → 3 + Sigmoid
```

The decoder reconstructs the output back to:

```text
[3, 96, 96]
```

The final `Sigmoid` activation keeps output pixel values between 0 and 1.

---

## 5. File Overview

The logical order of the files is:

```text
1. initialize.py
2. model.py
3. test_model_shape.py
4. train.py
5. visualize.py
6. denoise_custom.py
```

---

## 6. File Descriptions

### 6.1 initialize.py

This file is used to test the dataset loading and noise generation process.

It does the following:

```text
- Loads the STL10 dataset
- Converts images to tensors
- Creates a DataLoader
- Adds Gaussian noise to clean images
- Displays clean and noisy images
```

This file is useful for checking that the preprocessing and noise generation steps work correctly before training the model.

Run with:

```bash
python initialize.py
```

---

### 6.2 model.py

This file contains the neural network architecture.

It defines:

```text
DenoisingAutoencoder
```

The class contains:

```text
self.encoder
self.decoder
forward()
```

The `forward()` function sends the image through the encoder and decoder:

```python
encoded = self.encoder(x)
decoded = self.decoder(encoded)
return decoded
```

This file is imported by the training, visualization, shape testing, and custom denoising scripts.

---

### 6.3 test_model_shape.py

This file checks whether the model input and output shapes are correct.

It creates fake image data:

```python
dummy_input = torch.randn(4, 3, 96, 96)
```

Then it passes the dummy input through the model and prints:

```text
Input shape
Output shape
```

Expected output:

```text
Input shape: torch.Size([4, 3, 96, 96])
Output shape: torch.Size([4, 3, 96, 96])
```

This check is important because the denoising model output must have the same shape as the clean target image for `MSELoss` to work.

Run with:

```bash
python test_model_shape.py
```

---

### 6.4 train.py

This is the main training file.

It does the following:

```text
- Loads STL10 training and test data
- Adds Gaussian noise to clean images
- Sends noisy images into the autoencoder
- Compares model output with the clean image using MSELoss
- Updates model weights using Adam optimizer
- Tracks training and validation loss
- Saves the trained model
- Saves or displays the loss curve
```

Training configuration:

```text
Framework: PyTorch
Optimizer: Adam
Learning rate: 0.001
Batch size: 16
Epochs: 30
Noise factor: 0.08
Loss function: Mean Squared Error
Metric: Training and validation MSE loss
```

The trained model is saved as:

```text
STL10_denoising_autoencoder.pth
```

Run with:

```bash
python train.py
```

---

### 6.5 visualize.py

This file loads the trained model and visualizes its denoising performance on STL10 test images.

It does the following:

```text
- Loads the saved model
- Loads STL10 test images
- Adds Gaussian noise
- Passes noisy images through the trained model
- Displays noisy images, denoised outputs, and clean images
- Saves the visual comparison figure
```

The expected visual output has three rows:

```text
Noisy input
Denoised output
Clean target
```

The saved figure is:

```text
stl10_visual_results.png
```

Run with:

```bash
python visualize.py
```

---

### 6.6 denoise_custom.py

This file allows the trained model to denoise a custom external image.

It does the following:

```text
- Asks the user for an image path
- Loads the image
- Converts it to RGB
- Resizes it to 96 × 96
- Converts it to a tensor
- Passes it through the trained model
- Saves the denoised output image
- Saves a side-by-side comparison
```

The output images are saved in:

```text
outputs/
```

Important note:

```text
The custom image is resized to 96 × 96 because the model was trained on STL10 images with this resolution.
```

Run with:

```bash
python denoise_custom.py
```

---

## 7. Training Results

The model was trained for 30 epochs.

Final training results from the selected run:

```text
Final training loss: approximately 0.001882
Final validation loss: approximately 0.001852
```

The training and validation losses decreased consistently, and the final values stayed close to each other. This suggests that the model learned the denoising task without strong overfitting.

---

## 8. Output Files

The project may generate the following output files:

```text
STL10_denoising_autoencoder.pth
stl10_loss_curve.png
stl10_visual_results.png
outputs/
```

### STL10_denoising_autoencoder.pth

Saved trained model weights.

### stl10_loss_curve.png

Training and validation loss curve.

### stl10_visual_results.png

Visual comparison of noisy, denoised, and clean STL10 images.

### outputs/

Folder containing custom denoising outputs from `denoise_custom.py`.

---

## 9. How to Run the Project

Recommended order:

```bash
python initialize.py
python test_model_shape.py
python train.py
python visualize.py
python denoise_custom.py
```

If the trained model file already exists, it is not necessary to run `train.py` again before visualization or custom inference.

---

## 10. Requirements

The project requires Python and the following libraries:

```text
torch
torchvision
matplotlib
Pillow
```

A possible installation command is:

```bash
pip install torch torchvision matplotlib pillow
```

If using an NVIDIA GPU, a CUDA-enabled PyTorch installation is recommended.

---

## 11. Notes and Limitations

This project uses artificial Gaussian noise. Therefore, the model works best on images corrupted with a similar type and level of noise.

Main limitations:

```text
- Trained only on artificial Gaussian noise
- Uses a fixed noise factor of 0.08
- Custom images are resized to 96 × 96
- Some fine details may become smoothed
- The architecture is simple and does not use skip connections
```

Possible future improvements:

```text
- Test different noise levels
- Compare against classical filters such as Gaussian blur or median filtering
- Use U-Net-style skip connections to preserve fine details
- Train on real noisy-clean image pairs
- Support higher-resolution images using patch-based denoising
```

---

## 12. Project Summary

This project demonstrates a complete image denoising pipeline using a convolutional autoencoder. Clean STL10 images are artificially corrupted with Gaussian noise, and the model learns to reconstruct the original clean images. The project includes dataset preparation, model training, validation, visualization, and custom image inference.
