import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

from ResNet import B2_ResNet
import torchvision.models as models

class GATSegmentationModel(nn.Module):
    def __init__(self, training):
        super(GATSegmentationModel, self).__init__()

        self.resnet = B2_ResNet()

        self.training = training
        if self.training:
            self.initialize_weights()

        # self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        # self.conv2 = GATConv(hidden_dim * num_heads, 1, heads=1)  # Output layer with 1 channel

    def image_to_graph(self, input):
        # Reshape image into a vector
        x = input.view(-1, 1).float().numpy()

        # Compute nearest neighbors
        nn_model = NearestNeighbors(n_neighbors=4)  # Define the number of neighbors
        nn_model.fit(x)
        distances, indices = nn_model.kneighbors(x)

        # Creating edge_index from nearest neighbors indices
        edge_index = []
        for i in range(len(indices)):
            for j in range(1, len(indices[i])):
                edge_index.append((i, indices[i][j]))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Converting image to graph representation
        x = input.view(-1, 1)
        data = Data(x=x, edge_index=edge_index)

        return data

    def forward(self, input):
        x = self.resnet.conv1(input)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # x, edge_index = data.x, data.edge_index
        #
        # x = F.relu(self.conv1(x, edge_index))
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = self.conv2(x, edge_index)
        # x.view(-1, 256, 256)  # Reshape output to a 256x256 image

        return x

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

model = GATSegmentationModel(training=True)
tensor = torch.randn((2,3,256,256))
with torch.no_grad():
    out = model(tensor)
    print (out.shape)

print ('Done')