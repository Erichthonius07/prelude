# training/model.py
import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, num_features: int = 64, num_layers: int = 17):
        """
        A lightweight Denoising Convolutional Neural Network (DnCNN) optimized for mobile deployment.
        
        Args:
            in_channels: Number of color channels in the input image (3 for RGB).
            out_channels: Number of color channels in the output noise map (3 for RGB).
            num_features: Number of filters/channels in the hidden layers.
            num_layers: Total number of convolutional layers. 17 is standard for DnCNN.
        """
        super(DnCNN, self).__init__()
        
        layers = []
        
        # Layer 1: Initial feature extraction (Convolution + ReLU activation, no Batch Normalization)
        layers.append(nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Layers 2 to (num_layers - 1): Deep feature processing
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False))
            # Batch Normalization stabilizes the math during training
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))
            
        # Final Layer: Reconstruct the noise map (Convolution only)
        layers.append(nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1, bias=False))
        
        # Group everything into a single sequential pipeline
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Residual Learning.
        Instead of predicting the clean image, the network predicts the noise.
        """
        # The network extracts the noise (grain/artifacts) from the original image
        predicted_noise = self.network(x)
        
        # Subtract the predicted noise from the original image to get the clean result
        clean_image = x - predicted_noise
        
        return clean_image