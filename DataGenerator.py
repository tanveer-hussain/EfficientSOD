from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms as T
from PIL import Image
import torch
from torch import nn

class DatasetLoader(Dataset):


    def __init__(self, dir, d_type):

        self.x_path = os.path.join(dir, str(d_type), 'Images')

        self.y_path = os.path.join(dir, str(d_type), 'Labels')

        self.X = os.listdir(self.x_path)
        self.Y = os.listdir(self.y_path)

        self.length = len(self.X)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x_full_path = os.path.join(self.x_path, self.X[index])
        y_full_path = os.path.join(self.y_path, self.Y[index])

        x = Image.open(x_full_path).convert('RGB')
        y = Image.open(y_full_path).convert('L')

        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

        x = transform(x)
        y = transform(y)


        return x , y





