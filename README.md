# Floods-Detection-Using-Pre-Trained-EfficientNetV2B0
This project uses a U-Net with EfficientNetV2B0 as the encoder for image segmentation. It processes 12-channel inputs, applies upsampling, batch normalization, dropout, and L2 regularization. The model trains with Adam, binary cross-entropy loss, and early stopping. A post-processing step converts predictions into binary masks.
