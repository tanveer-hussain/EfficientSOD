import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class GATSegmentationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(GATSegmentationModel, self).__init__()

        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, 1, heads=1)  # Output layer with 1 channel

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return x.view(-1, 256, 256)  # Reshape output to a 256x256 image

# Generating an adjacency matrix for a grid-like graph
rows, cols = 256, 256  # Image dimensions
num_nodes = rows * cols

edge_index = []

for i in range(rows):
    for j in range(cols):
        node_id = i * cols + j
        if j < cols - 1:
            edge_index.append((node_id, node_id + 1))
        if i < rows - 1:
            edge_index.append((node_id, node_id + cols))

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

# Generating a random image tensor (256x256)
random_image = torch.rand((256, 256))

# Converting image to graph representation
x = random_image.view(-1, 1)
data = Data(x=x, edge_index=edge_index)

# Model initialization and inference
input_dim = 1  # Single-channel image
hidden_dim = 64
num_heads = 4

model = GATSegmentationModel(input_dim, hidden_dim, num_heads)
output = model(data)
print(output.shape)  # Output shape (batch_size, height, width)
