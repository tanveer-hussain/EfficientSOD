import torch
from torch import nn, cuda
from DataGenerator import DatasetLoader
from ModelNetworks import BaseNetwork_3
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, opt, crit, train_loader, epoch):
    model.train()
    for i, (X, Y) in enumerate(train_loader):
        X = X.to('cuda')
        Y = Y.to('cuda')
    
        output = model(X)

        loss = crit(output, Y) # this is for loss comparison
    
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()

def main():

    print('__Number of CUDA Devices:', cuda.device_count(), ', active:', cuda.current_device())
    print ('Device name: .... ', cuda.get_device_name(cuda.current_device()), ', available >', cuda.is_available())

    model = BaseNetwork_3.DenseNetBackbone()

    cudnn.benchmark = True
    model.to('cuda')
    print(count_parameters(model))

    base_lr = 0.0001
    epochs = 100
    weight_decay = 1e-3
    k = 0
    total_loss = []

    optimizerr = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    criterion = nn.MSELoss().to('cuda')
    
    print('Model on GPU: ', next(model.parameters()).is_cuda)

    dataset_path = r'D:\My Research\Video Summarization\VS via Saliency\SIP'
    d_type = ['Train', 'Test']

    train_data = DatasetLoader(dataset_path, d_type[0])
    train_loader = DataLoader(train_data, batch_size=48, shuffle=True, num_workers=16, drop_last=True)

    test_data = DatasetLoader(dataset_path, d_type[1])
    for epoch in range(0, epochs):

        current_loss = train(model, optimizerr, criterion, train_loader, epoch)
        if epoch%4 == 0:
            print("Epoch: %d of: %d, loss: %f"%(epoch,epochs, current_loss))
        total_loss.append(current_loss)
    
    torch.save(model, 'TrainedModels\\DDNet_500Model.pt')
    torch.save(model.state_dict(), 'TrainedModels\\DDNet_500Weights.pt')

if __name__ == '__main__':
    main()
