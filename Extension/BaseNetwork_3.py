from torch import nn
import torch
import torch.nn.functional as F

class DenseNetBackbone(nn.Module):
    def __init__(self):
        super(DenseNetBackbone, self).__init__()

        # ******************** Encoding image ********************

        

        # ******************** Decoding image ********************
        

    def forward(self, x):
        pass


        return x

device = torch.device('cuda' if torch.cuda.is_available else "cpu")

x = torch.randn((2, 3, 224, 224)).to(device)
depth = torch.randn((2, 3, 224, 224)).to(device)
# # gt = torch.randn((12, 1, 224, 224)).to(device)
model = DenseNetBackbone().to(device)
y = model(x,depth)
print (y.shape)
