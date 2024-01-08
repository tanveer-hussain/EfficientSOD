import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class GATSegmentationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads):
        super(GATSegmentationModel, self).__init__()

        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, num_classes, heads=1)  # Output layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# Create a random 256x256 image tensor
random_image = torch.rand((256, 256))

# Convert the image to a graph representation
# For simplicity, let's create a grid-like graph where each pixel is a node connected to its neighbors
rows, cols = random_image.shape
num_nodes = rows * cols
edges = []  # Store edges between nodes


# Define a function to add edges between neighboring pixels
def add_edge(i, j):
    edges.extend([(i * cols + j, (i + ni) * cols + (j + nj)) for ni in [-1, 0, 1] for nj in [-1, 0, 1] if
                  0 <= i + ni < rows and 0 <= j + nj < cols])


for i in range(rows):
    for j in range(cols):
        add_edge(i, j)

# Create a PyTorch Geometric Data object for the graph
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
x = random_image.view(-1, 1)  # Flatten the image into a single channel

data = Data(x=x, edge_index=edge_index)

# Create an instance of the segmentation model
input_dim = 1  # Input dimension (single-channel image)
hidden_dim = 64
num_classes = 2  # Example: binary segmentation
num_heads = 4

model = GATSegmentationModel(input_dim, hidden_dim, num_classes, num_heads)

# Pass the image through the model
output = model(data)
print(output.shape)  # Output shape (num_nodes, num_classes)