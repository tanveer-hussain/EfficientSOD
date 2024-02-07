import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("./download.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device) # textual description of what you're interested.

import torch
import torch.nn as nn


class TransformerDecoder(nn.Module):
    def __init__(self, input_channels, output_channels, target_size):
        super(TransformerDecoder, self).__init__()

        # Transposed convolution layers for upsampling
        self.transposed_conv1 = nn.ConvTranspose2d(input_channels, 512, kernel_size=4, stride=2, padding=1)
        self.transposed_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.transposed_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.transposed_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.transposed_conv5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)

        # Transformer layers
        self.transformer1 = nn.Transformer(d_model=32, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

        # Final convolution layer to obtain the target size
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1)

        # Target size for the output tensor
        self.target_size = target_size

    def forward(self, x):
        # Upsampling using transposed convolution layers
        x = self.transposed_conv1(x)
        x = self.transposed_conv2(x)
        x = self.transposed_conv3(x)
        x = self.transposed_conv4(x)
        x = self.transposed_conv5(x)

        # Resize to the target size using bilinear interpolation
        x = nn.functional.interpolate(x, size=self.target_size, mode='bilinear', align_corners=False)

        # Transformer layer
        x = x.view(self.target_size[0], self.target_size[1], -1)  # Reshape for transformer
        x = self.transformer1(x, x)
        x = x.view(1, 32, self.target_size[0], self.target_size[1])  # Reshape back to the original shape

        # Final convolution layer
        x = self.final_conv(x)

        return x


# Instantiate the decoder
decoder = TransformerDecoder(input_channels=768, output_channels=1, target_size=(224, 224))

# Create a dummy input tensor with size [1, 768, 7, 7]
# dummy_input = torch.randn(1, 768, 7, 7)
#
# # Forward pass through the decoder
# output = decoder(dummy_input)
# print("Output size:", output.size())

with torch.no_grad():
    image_features, temp = model.encode_image(image)
    print (temp.shape)
    # text_features = model.encode_text(text)
    #
    # logits_per_image, logits_per_text, temp = model(image, text)
    # # print (temp.shape)
    # saliency = decoder(temp)
    # print (saliency.shape)
    # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]