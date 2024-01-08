import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

from ResNet import B2_ResNet
class GATSegmentationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(GATSegmentationModel, self).__init__()

        self.resnet = B2_ResNet()

        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, 1, heads=1)  # Output layer with 1 channel

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x.view(-1, 256, 256)  # Reshape output to a 256x256 image


def image_to_graph(image):
    # Reshape image into a vector
    x = image.view(-1, 1).float().numpy()

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
    x = image.view(-1, 1)
    data = Data(x=x, edge_index=edge_index)

    return data

# Generating a random image tensor (256x256)
random_image = torch.rand((256, 256))
data = image_to_graph(random_image)

# Model initialization and inference
input_dim = 1  # Single-channel image
hidden_dim = 64
num_heads = 4

model = GATSegmentationModel(input_dim, hidden_dim, num_heads)
output = model(data)
print(output.shape)  # Output shape (batch_size, height, width)
