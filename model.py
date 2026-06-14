import torch
import torch.nn as nn


#create custom pytorch  mdel
class DenoisingAutoencoder(nn.Module):

    def __init__(self):
        #initialize parent nn.Module
        super().__init__()

        # Encoder: compress image features
        self.encoder = nn.Sequential(
            # input: [batch, 3, 96, 96]
            #input channel=3, output channel=32,kernel size=3x3, padding=1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            #Rectified linear unit activation function creates non linearity
            #changes negative values into zero
            nn.ReLU(),

            # takes 32 feature maps, produces 64: [batch, 32, 96, 96] -> [batch, 64, 96, 96]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            #reduces widht and height b half, why?
            #compresses the image, forces it to learn general structure rather than memorizing every pixel
            nn.MaxPool2d(2, 2),  # [batch, 64, 48, 48]
            # Learns deeper image features: [batch, 64, 48, 48] -> [batch, 128, 48, 48]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            #halves the size
            nn.MaxPool2d(2, 2),  # [batch, 128, 24, 24]
        )

        # Decoder: reconstruct image from compressed features back to 96x96
        self.decoder = nn.Sequential(
            #upsample the feature map
            nn.ConvTranspose2d(
                128, 64,
                kernel_size=2,
                stride=2
            ),  # [batch, 128, 24, 24] -> [batch, 64, 48, 48]
            # add non linearity
            nn.ReLU(),

            #refine the reconstructed features after upsampling features
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            #upsample again
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=2,
                stride=2
            ),  # [batch, 64, 48, 48] -> [batch, 32, 96, 96]
            nn.ReLU(),

            #Reduce channel back to 3, the output must be RGB
            #input and output image shape must match to calculate the loss 
            nn.Conv2d(32, 3, kernel_size=3, padding=1), #[batch, 3, 96, 96]

            # output pixel values between 0 and 1 (Matches ToTensor())
            nn.Sigmoid()
        )

#Noisy image -> encoder -> bottleneck -> decoder -> return denoised image
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded