from torch import nn
import torchvision
from deformable_conv import DeformConv2d
import torch
import torch.nn.functional as F

class DenseNetBackbone(nn.Module):
    def __init__(self):
        super(DenseNetBackbone, self).__init__()

        # ******************** Encoding image ********************

        originalmodel = torchvision.models.densenet169(pretrained=False, progress=True)
        # pretrained_model = models.vgg16(pretrained=True).features
        self.custom_model = nn.Sequential(*list(originalmodel.features.children())[:-5])
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),

        )

        self.deform1 = DeformConv2d(256, 128, 3, padding=1, modulation=True)
        self.deform2 = DeformConv2d(128, 128, 3, padding=1, modulation=True)
        # nn.ReLU(),
        self.deform3 = DeformConv2d(128, 64, 3, padding=1, modulation=True)

        # ******************** Decoding image ********************
        self.deconv1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2, padding=1)
        #self.deform = DeformConv2d(256, 128, 3, padding=1, modulation=True)
        self.upsampling1 = nn.Upsample(size=(112, 112), mode='bilinear', align_corners=True)

        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(3, 3), stride=2, padding=1)
        self.upsampling2 = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)



    def forward(self, x):
        # ******************** Initializing filters ********************
        # h_filter = torch.tensor([[-1., 0., 1.],
        #                 [-2., 1., 2.],
        #                 [-1., 0., 1.]]).to('cuda')
        # h_filter = h_filter.view(1, 1, 3, 3).repeat(256, 256, 1, 1) # convolution mask (gx)
        #
        # v_filter = torch.tensor([[-1., -2., -1.],
        #                          [0., 0., 0.],
        #                          [1., 2., 1.]]).to('cuda')
        # v_filter = v_filter.view(1, 1, 3, 3).repeat(1, 1, 1, 1) # convolution mask (gy)
        # ******************** Encoding image ********************
        x = self.custom_model(x)
        x = self.layer1(x)

        deform1_x = self.deform1(x)
        deform2_x = self.deform2(deform1_x)
        x = self.deform3(deform1_x)
        x = self.deform3(deform2_x)



        # # ******************** Decoding image ********************
        x = self.deconv1(x)
        x = self.deconv2(x)
        # x = self.deform(x)
        # x = F.conv2d(x, h_filter)

        x = self.upsampling1(x)
        x = self.deconv3(x)
        # x = F.conv2d(x, v_filter)
        x = self.upsampling2(x)
        x = self.upsampling2(x)


        return x
