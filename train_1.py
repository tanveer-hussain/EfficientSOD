import torch
from DataGenerator import DatasetLoader
from ModelNetworks import BaseNetwork_3
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from torch import nn, cuda
import numpy as np
import winsound
import timeit
import os
from scipy.ndimage import gaussian_filter

# ssim_loss = pytorch_ssim.SSIM()
freq = 500
dur = 100
kernel = np.ones((3,3), np.uint8)
def g_difference(output,channels_in, channels_out):
    diffs = gaussian_filter(output.cpu().detach().numpy(), sigma=3) - gaussian_filter(
        output.cpu().detach().numpy(), sigma=5)
    conv2d = nn.Conv2d(channels_in, channels_out, 3, bias=False)
    with torch.no_grad():
        conv2d.weight = torch.nn.Parameter(torch.FloatTensor(diffs).cuda())
    return conv2d

def train(model, opt, criterion, train_loader, epoch):
    model.train()
    for i, (X, Y) in enumerate(train_loader):


        opt.zero_grad()
        X = X.to('cuda')
        Y = Y.to('cuda')
        output = model(X)

        if epoch == 499:
            winsound.Beep(freq, dur)

        #
        # for image_number in range(0,list(Y.shape)[0]-10):
        #     single_image = output[image_number,:,:,:]
        #     single_image = single_image.permute(1,2,0).detach().cpu().numpy()
        #
        #     modified_image = cv2.erode(single_image, kernel, iterations=1)
        #     modified_image = cv2.dilate(modified_image, kernel, iterations=1)
        #
        #     modified_image_T = torch.from_numpy(modified_image).to('cuda')
        #     output[image_number,:,:,:] = modified_image_T

        #     if epoch == 30:
        #         winsound.Beep(freq, dur)
        #         cv2.imshow(str(image_number), single_image)
        #         cv2.waitKey(50)
        #
        # if epoch == 38:
        #     cv2.destroyAllWindows()

        loss = criterion(output,Y)
        #
        if i % 200 == 0:
            print ('Epoch: ',epoch,',Loss: ',round(loss.item(),4))


        loss.backward()
        # ssim_out.backward()
        opt.step()

        # return loss

def main():

    # model = BaseNetwork_3.myModel()
    model = BaseNetwork_3.DenseNetBackbone()
    cudnn.benchmark = True
    model.to('cuda')
    # model.state_dict(torch.load('TrainedModels\\SIP_2Dense_Deform_ERDIil_TLearning.pt'))
    summary(model, (3, 224,224))

    # datasets = SIP  , DUT-RGBD  , NLPR  , NJU2K

    dataset_name = 'SIP'

    base_lr = 0.0001
    epochs = 500
    weight_decay = 1e-3
    optimizerr = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))

    dataset_path = r'D:\My Research\Datasets\Saliency Detection\\' + dataset_name

    d_type = ['Train', 'Test']

    criterion = nn.MSELoss().to('cuda')
    # criterion = nn.BCEWithLogitsLoss().to('cuda')
    # criterion = pytorch_ssim.SSIM()

    train_data = DatasetLoader(dataset_path, d_type[0])
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=8, drop_last=True)

    start_time = timeit.default_timer()
    for epoch in range(0, epochs):

        train(model, optimizerr, criterion, train_loader, epoch)
        if epoch % 4 ==0:

            print("Epoch: %d of %d, loss "%(epoch,epochs))
    print ('Time consumed during training >> , ', timeit.default_timer() -start_time , ' seconds')
    model_name = dataset_name  + '20PercentSSODDDNet-500.pt'
    weights_name = dataset_name + '20PercentSSODDDNet-500.pt'
    model_path = os.path.join('TrainedModels' , model_name)
    weights_path = os.path.join('TrainedModels' , weights_name)
    torch.save(model.state_dict(), weights_path)
    torch.save(model, model_path)
    print ('Model Saved..!')

if __name__ == '__main__':
    main()
