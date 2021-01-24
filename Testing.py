import torch
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np
import cv2
import timeit


# datasets = SIP  , DUT-RGBD  , NLPR  , NJU2K
model_path = os.path.join('TrainedModels\\DDNet_500Model.pt')
model = torch.load(model_path)
model.eval()
kernel = np.ones((5,5), np.uint8)

def preprocess_image(img):
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    x = transform(img)
    x = torch.unsqueeze(x, 0)
    x = x.cuda(0)
    return x
def predictions(img):

    x = preprocess_image(img)
    start_time = timeit.default_timer()
    output = model(x)
    output = torch.squeeze(output, 0)

    output = output.detach().cpu().numpy()
    output = output.dot(255)
    output *= output.max()/255.0
    # print (max(output))
    # output = cv2.erode(output, kernel, iterations=2)
    # output = cv2.dilate(output, kernel, iterations=1)
    return output

def testing_code_dir(input_dir, output_dir):

    val_base_path_images = os.listdir(input_dir)
    for single_image in val_base_path_images:
        full_path = input_dir + single_image

        img = Image.open(full_path).convert("RGB")

        output = predictions(img)
        output = np.transpose(output, (1, 2, 0))
        # cv2.imshow('', output)
        # cv2.waitKey(50)

        output_path = output_dir + single_image[0:(len(single_image) - 3)] + "png"
        cv2.imwrite(output_path, output)
        print("Reading: %s\n writing: %s " % (full_path, output_path))

# # testing code SIP
testing_code_dir(r'D:\My Research\Datasets\Saliency Detection\SIP\Test\Images\\',r'C:\Users\user02\Documents\GitHub\EfficientSOD\SIPTestResults\\')
