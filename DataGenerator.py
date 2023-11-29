from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms as T
from PIL import Image
import torch
from torch import nn
import imageio.v3 as io
import numpy as np
from matplotlib import cm
import cv2
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
        color_coverted = cv2.imread(y_full_path, cv2.IMREAD_GRAYSCALE)
        y = Image.fromarray(color_coverted)#.convert('L')
        # y = Image.open(y_full_path).convert('L')
        # y = io.imread(y_full_path)
        # y = Image.fromarray(np.uint8(cm.gist_earth(y)*255))

        transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

        x = transform(x)
        y = transform(y)

        # print (x.shape, '\t' , y.shape, '\t Reading...', y_full_path)



        return x , y





