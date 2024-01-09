import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from itertools import product

from ResNet import B2_ResNet
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GATSegmentationModel(nn.Module):
    def __init__(self, training):
        super(GATSegmentationModel, self).__init__()

        self.resnet = B2_ResNet()

        self.training = training
        if self.training:
            self.initialize_weights()

        self.conv1 = GATConv(64, 32, heads=4)
        self.conv2 = GATConv(32 * 4, 1, heads=1)  # Output layer with 1 channel

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

        data = self.image_to_graph(x, radius=5, size=64).to(device)  # 64x64x64 > x=[524288,1], edge_index=[2,1572864]

        x, edge_index = data.x, data.edge_index
        #
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        y = x.view(-1, 64, 64)  # Reshape output to a 256x256 image

        return y

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