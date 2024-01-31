import torch
from torchvision import transforms as T
from PIL import Image
import os
from SODModel import Generator
import numpy as np
import cv2
import timeit


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_image(img):
    transform = T.Compose([T.Resize((352, 352)), T.ToTensor()])
    x = transform(img)
    x = torch.unsqueeze(x, 0)
    x = x.to(device)
    return x
def predictions(img, depth, h , w):

    x = preprocess_image(img)
    depth = preprocess_image(depth)

    start_time = timeit.default_timer()
    output = generator.forward(x, depth, training=False)
    print (h, w)
    # output = output[0]
    output = torch.nn.functional.upsample(output, size=(h, w), mode='bilinear', align_corners=True)
    output = torch.squeeze(output, 0).sigmoid()

    output = output.detach().cpu().numpy()
    output = output.dot(255)
    output *= output.max()/255.0

    return output

def testing_code_dir(input_dir, output_dir):

    val_base_path_images = os.listdir(os.path.join(input_dir,'Images'))
    for single_image in val_base_path_images:
        image_path = input_dir + '/Images/' + single_image
        depth_path = input_dir + '/Depth/' + single_image[0:(len(single_image) - 3)] + "png"

        img = Image.open(image_path).convert("RGB")
        depth = Image.open(depth_path).convert('RGB')

        w, h = img.size

        output = predictions(img, depth, h , w)
        output = np.transpose(output, (1, 2, 0))
        # cv2.imshow('', output)
        # cv2.waitKey(50)

        output_path = output_dir + single_image[0:(len(single_image) - 3)] + "png"
        cv2.imwrite(output_path, output)
        print("Reading: %s\n writing: %s " % (image_path, output_path))

datasets = ['SIP', 'DUT-RGBD', 'NLPR', 'NJU2K']
current_dataset = datasets[1]
dataset_path = r'/home/hussaint/SODDatasets/' + current_dataset
d_type = ['Train', 'Test']
# # testing code SIP
input_dir = dataset_path + '/Test/'
output_dir = r'Output/'

# datasets = SIP  , DUT-RGBD  , NLPR  , NJU2K
weights_file = 'DDNetWts_' + current_dataset + '_99.pt'


generator = Generator(channel=32, latent_dim=3).to(device)
print (generator.load_state_dict(torch.load(os.path.join('/home/hussaint/EfficientSOD/JournalExtension/', weights_file))))

# model_path = os.path.join('/home/hussaint/EfficientSOD/JournalExtension/DDNetDUT-RGBD_99.pt')
# model = torch.load(model_path)
# model.eval().to(device)
print ('Model Loaded...')

testing_code_dir(input_dir,output_dir)
