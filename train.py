import torch
from torch import nn, cuda
from DataGenerator import DatasetLoader
from ModelNetworks import BaseNetwork_3
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

def train(model, opt, crit, train_loader, epoch):
    model.train()
    for i, (X, Y) in enumerate(train_loader):
        X = X.to('cuda')
        #print('X in train model is on GPU: ', X.is_cuda)
        Y = Y.to('cuda')
        #print('Y in train model is on GPU: ', Y.is_cuda)

        output = model(X)

        loss = crit(output, Y)
        if i % 10 == 0:
            print ('Loss: ',loss)

        opt.zero_grad()
        loss.backward()
        opt.step()


def main():

    print('__Number of CUDA Devices:', cuda.device_count(), ', active:', cuda.current_device())
    print ('Device name: .... ', cuda.get_device_name(cuda.current_device()), ', available >', cuda.is_available())

    model = BaseNetwork_3.DenseNetBackbone()

    cudnn.benchmark = True
    model.to('cuda')
    summary(model, (3, 236,236))

    base_lr = 0.0001
    epochs = 10
    weight_decay = 1e-3
    k = 0

    optimizerr = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    criterion = nn.MSELoss().to('cuda')
    
    print(next(model.parameters()).is_cuda)

    dataset_path = r'D:\My Research\Video Summarization\VS via Saliency\SIP'
    d_type = ['Train', 'Test']

    train_data = DatasetLoader(dataset_path, d_type[0])
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=8, drop_last=True)

    test_data = DatasetLoader(dataset_path, d_type[1])
    #test_loader = DataLoader(test_data, 16, shuffle=False, num_workers=4, drop_last=True)

    for epoch in range(0, epochs):

        train(model, optimizerr, criterion, train_loader, epoch)
        print("Epoch: %d, of epochs: %d"%(epoch,epochs))

    torch.save(model, 'TrainedModels\\DDNet_500Model.pt')

if __name__ == '__main__':
    main()
