import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt

class SimpleGATSegmentationModel(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_heads=4):
        super(SimpleGATSegmentationModel, self).__init__()

        # GATConv layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1)

    def forward(self, data):
        # Apply GATConv layers
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = self.conv2(x, data.edge_index)

        # Apply softmax to get probabilities
        x = F.softmax(x, dim=1)

        return x

# Create a dynamic graph using torch_geometric.data.Data
num_nodes = 256
edge_index = torch.randint(0, num_nodes, (2, num_nodes * 4), dtype=torch.long)
edge_index, _ = torch.unique(edge_index, dim=1, return_inverse=True)
edge_index = torch.stack([edge_index[0], edge_index[1]])

x = torch.randn(num_nodes, 3)  # Node features (e.g., RGB values)
y = torch.randint(0, 21, (num_nodes,))  # Node labels

data = Data(x=x, y=y, edge_index=edge_index)

# Create an instance of the model
in_channels = 3  # Input channels (e.g., for RGB images)
out_channels = 21  # Number of classes (adjust based on your dataset)
model = SimpleGATSegmentationModel(in_channels, out_channels)

# Pass the input through the model
with torch.no_grad():
    output = model(data)
    print (output.shape)

# Display the segmentation map
# plt.imshow(output.squeeze().argmax(dim=1).cpu(), cmap='viridis')
# plt.title('Segmentation Map')
# plt.show()
