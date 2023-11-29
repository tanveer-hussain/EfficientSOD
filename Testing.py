import torch
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np
import cv2
import timeit


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# datasets = SIP  , DUT-RGBD  , NLPR  , NJU2K
model_path = os.path.join('DDNet_100.pt')
model = torch.load(model_path)
model.eval().to(device)

def preprocess_image(img):
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    x = transform(img)
    x = torch.unsqueeze(x, 0)
    x = x.to(device)
    return x
def predictions(img, h , w):

    x = preprocess_image(img)
    start_time = timeit.default_timer()
    output = model(x)
    print (h, w)
    # output = output[0]
    output = torch.nn.functional.upsample(output, size=(h, w), mode='bilinear', align_corners=True)
    output = torch.squeeze(output, 0).sigmoid()

    output = output.detach().cpu().numpy()
    output = output.dot(255)
    output *= output.max()/255.0

    return output

def testing_code_dir(input_dir, output_dir):

    val_base_path_images = os.listdir(input_dir)
    for single_image in val_base_path_images:
        full_path = input_dir + single_image

        img = Image.open(full_path).convert("RGB")

        w, h = img.size

        output = predictions(img, h , w)
        output = np.transpose(output, (1, 2, 0))
        # cv2.imshow('', output)
        # cv2.waitKey(50)

        output_path = output_dir + single_image[0:(len(single_image) - 3)] + "png"
        cv2.imwrite(output_path, output)
        print("Reading: %s\n writing: %s " % (full_path, output_path))

# # testing code SIP
input_dir = r'/home/hussaint/SODDatasets/NLPR/Test/Images/'
output_dir = r'Output/'
testing_code_dir(input_dir,output_dir)
