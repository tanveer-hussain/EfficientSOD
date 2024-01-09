import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from itertools import product

from ResNet import B2_ResNet
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(ChannelReducer, self).__init__()

        # 1x1 Convolution layers to reduce channels gradually
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

        # Channel pooling to further reduce channels
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Channel attention blocks
        self.se1 = self._make_se_block(in_channels // 2, reduction_ratio)
        self.se2 = self._make_se_block(in_channels // 4, reduction_ratio)
        self.se3 = self._make_se_block(out_channels, reduction_ratio)
        self.se4 = self._make_se_block(out_channels, reduction_ratio)

    def _make_se_block(self, in_channels, reduction_ratio):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply 1x1 convolutions
        x1 = torch.relu(self.conv1(x))
        x1 = x1 * self.se1(x1)  # Apply channel attention

        x2 = torch.relu(self.conv2(x1))
        x2 = x2 * self.se2(x2)  # Apply channel attention

        x3 = torch.relu(self.conv3(x2))
        x3 = x3 * self.se3(x3)  # Apply channel attention

        # Channel pooling to reduce spatial dimensions to 1x1
        # x3 = self.pool(x3)

        return x3  # Remove singleton spatial dimensions


class WeightedFusionAttentionCNN(nn.Module):
    def __init__(self):
        super(WeightedFusionAttentionCNN, self).__init__()

        # Upsampling layers to match the final size
        self.up1 = nn.Upsample(size=(32, 32), mode='bilinear')
        self.up2 = nn.Upsample(size=(64, 64), mode='bilinear')

        # Convolutional layers for each input
        self.conv1 = nn.Conv2d(768, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),  # Adjust channels for attention
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Final convolutional layer for output
        self.final_conv = nn.Conv2d(256, 1, kernel_size=3, padding=1)

    def forward(self, x1, x2, x3):
        # Upsample smaller inputs to match the size of the largest one
        x2 = self.up1(x2)
        x3 = self.up2(x3)

        # Apply convolutional layers to each input
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        # Concatenate the feature maps
        fused = torch.cat((x1, x2, x3), dim=1)

        # Apply attention mechanism
        attention_weights = self.attention(fused)

        # Apply attention to fused features
        fused_attention = fused * attention_weights

        # Final convolutional layer for output
        output = self.final_conv(fused_attention)

        return output

class GATSegmentationModel(nn.Module):
    def __init__(self, training):
        super(GATSegmentationModel, self).__init__()

        self.resnet = B2_ResNet()

        self.training = training
        if self.training:
            self.initialize_weights()

        self.conv2_reduce = ChannelReducer(512, 64)
        self.conv3_reduce = ChannelReducer(1024, 64)
        self.conv4_reduce = ChannelReducer(2048, 64)

        self.gatconv21 = GATConv(32, 16, heads=4)
        self.gatconv22 = GATConv(16 * 4, 8, heads=4)

        self.gatconv31 = GATConv(16, 8, heads=8)
        self.gatconv32 = GATConv(8 * 8, 8, heads=8)

        self.gatconv41 = GATConv(8, 4, heads=12)
        self.gatconv42 = GATConv(4 * 12, 4, heads=12)

        self.wghted_attn = WeightedFusionAttentionCNN()

    def image_to_graph(self, input, radius, size):
        height, width = input.shape[-2], input.shape[-1]
        num_nodes = height * width

        # Constructing edge indices by considering connections within a radius
        edge_index = []
        for i in range(height):
            for j in range(width):
                node_idx = i * width + j
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbor_idx = ni * width + nj
                            if node_idx != neighbor_idx:  # Exclude self-loops
                                edge_index.append((node_idx, neighbor_idx))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)

        # Ensure each edge exists in both directions (undirected graph)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        x = input.view(-1, size)  # Assuming the input is shaped for the ResNet
        data = Data(x=x, edge_index=edge_index)

        return data

    def forward(self, input):
        x = self.resnet.conv1(input)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4(x3)  # 2048 x 8 x 8

        x2 = self.conv2_reduce(x2)
        x3 = self.conv3_reduce(x3)
        x4 = self.conv4_reduce(x4)

        data2 = self.image_to_graph(x2, radius=5, size=32).to(device)  # 64x64x64 > x=[524288,1], edge_index=[2,1572864]
        x2, edge_index2 = data2.x, data2.edge_index
        x2 = F.relu(self.gatconv21(x2, edge_index2))
        x2 = F.dropout(x2, p=0.5, training=self.training)
        x2 = F.relu(self.gatconv22(x2, edge_index2))
        y2 = x2.view(-1, 32, 32)

        data3 = self.image_to_graph(x3, radius=5, size=16).to(device)  # 64x64x64 > x=[524288,1], edge_index=[2,1572864]
        x3, edge_index3 = data3.x, data3.edge_index
        x3 = F.relu(self.gatconv31(x3, edge_index3))
        x3 = F.dropout(x3, p=0.5, training=self.training)
        x3 = F.relu(self.gatconv32(x3, edge_index3))
        y3 = x3.view(-1, 16, 16)

        data4 = self.image_to_graph(x4, radius=5, size=8).to(device)  # 64x64x64 > x=[524288,1], edge_index=[2,1572864]
        x4, edge_index4 = data4.x, data4.edge_index
        x4 = F.relu(self.gatconv41(x4, edge_index4))
        x4 = F.dropout(x4, p=0.5, training=self.training)
        x4 = F.relu(self.gatconv42(x4, edge_index4))
        y4 = x4.view(-1, 8, 8)
        print(y4.shape, y3.shape, y2.shape)

        y = self.wghted_attn(x4, x3, x2)
        print (y.shape)


        return y2

    def initialize_weights(self):
        print('Loading weights...')
        res50 = models.resnet101(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        print(self.resnet.load_state_dict(all_params))


# from ptflops import get_model_complexity_info

# with torch.cuda.device(0):
#   net = GATSegmentationModel(training=False)
#   x = torch.randn((3, 224, 224))
#   macs, params = get_model_complexity_info(net,(3, 256, 256), as_strings=True,
#                                            print_per_layer_stat=True, verbose=True)
#   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#   print('{:<30}  {:<8}'.format('Number of parameters: ', params))


model = GATSegmentationModel(training=True).to(device)
tensor = torch.randn((2, 3, 256, 256)).to(device)

with torch.no_grad():
    out = model(tensor).to(device)
    print(out.shape)

print('Done')