import torch
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np
import cv2
import timeit


# datasets = SIP  , DUT-RGBD  , NLPR  , NJU2K
dataset_name = 'SIP'
model_name = dataset_name  + 'DDNet-1000.pt'
model_path = os.path.join('TrainedModels' , model_name)
model = torch.load(model_path)
model.eval()
kernel = np.ones((5,5), np.uint8)
# img = Image.open('19.jpg').convert("RGB")


def preprocess_image(img):
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    x = transform(img)
    x = torch.unsqueeze(x, 0)
    x = x.cuda(0)
    return x

#
def predictions(img):

    x = preprocess_image(img)
    start_time = timeit.default_timer()
    output = model(x)
    output = torch.squeeze(output, 0)
    # print('Single prediction time consumed >> , ', timeit.default_timer() - start_time, ' seconds')

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
#
# # testing code SIP
testing_code_dir(r'D:\My Research\Datasets\Saliency Detection\SIP\Test\Images\\',r'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\SIP Test Data Results\Temp\\')

# testing NLPR
# testing_code_dir(r'D:\My Research\Datasets\Saliency Detection\NLPR\Test\Images\\',r'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\NLPR Data Results\Temp\\')

# testing Chokepoint dataset
# testing_code_dir(r'D:\My Research\Datasets\Saliency Detection\SOD_Dataset\Images\\',r'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\CrossData\Output\\')

# testing NJU2K
# testing_code_dir(r'D:\My Research\Datasets\Saliency Detection\NJU2K\Test\Images\\',r'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\NJU2K Data Results\Temp\\')

# testing code DUTRGBD
# testing_code_dir(r'D:\My Research\Datasets\Saliency Detection\DUT-RGBD\Test\Images\\',r'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\DUTRGB-D Test Data Results\Temp\\')
#


# def single_image(single_image_path):
#     img = Image.open(single_image_path).convert('RGB')
#     output = predictions(img).numpy()
#     output = np.transpose(output, (1, 2, 0))
#
#     print(output.shape)
#
#     cv2.imwrite('temp.png',output)



import matplotlib.pyplot as plt
#single_image(r'D:\My Research\Video Summarization\VS via Saliency\SIP\Test\Images\18.jpg')
# def main():
#     model = torch.load('deformable_saliency_1000_mse_dropout.pt')
#     model.eval()
#     x_full_path = r'D:\My Research\Video Summarization\VS via Saliency\SIP\Test\Images\18.jpg'
#     x = Image.open(x_full_path).convert('RGB')
#     transform = T.Compose([T.Resize((224, 224)),T.ToTensor()])
#
#     x = transform(x)
#     x = torch.unsqueeze(x, 0)
#     x = x.cuda(0)
#
#     y = model(x)
#     print(y.shape)
#     y = torch.squeeze(y, 0)
#     y = y.detach().cpu()
#
#     print ('xxx >', x.shape)
#     y = np.transpose(y, (1, 2, 0))
#     print('yyy >', x.shape)
#     # y = T.ToPILImage()(y)
#
#     plt.imshow(y)
#     plt.show()
#
#     return y

# y = main()