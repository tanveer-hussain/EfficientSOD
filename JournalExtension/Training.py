import torch
from torch import nn, cuda
from DataGenerator import DatasetLoader
from ModelNetworks import SODModel
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.nn.functional as F


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OhemCrossEntropy2dTensor(nn.Module):
    def __init__(self, ignore_label, reduction='elementwise_mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)


class CriterionOhemDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the models.
    '''

    def __init__(self, ignore_index=255, thresh=0.7, min_kept=100000, reduce=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh=thresh, min_kept=min_kept)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduce=reduce)
        if not reduce:
            print("disabled the reduce.")

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)

        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)

        print(scale_pred.shape, ', pred \t', target.shape, ', target <')

        loss1 = self.criterion1(scale_pred, target)

        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        loss2 = self.criterion2(scale_pred, target)

        return loss1 + loss2 * 0.4


class CriterionDSN(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=255, reduce=True):
        super(CriterionDSN, self).__init__()

        self.ignore_index = ignore_index
        self.reduce = reduce

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = preds[0]
        scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)

        loss1 = super(CriterionDSN, self).forward(scale_pred, target)
        scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        scale_pred = preds[1]
        loss2 = super(CriterionDSN, self).forward(scale_pred, target)

        return loss1 + loss2 * 0.4


from scipy import misc


def visualize_uncertainty_prior_init(var_map):
    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_prior_int.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)


def visualize_gt(var_map):
    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)


CE = torch.nn.BCELoss().to(device)
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True).to(device)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).sum()


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


import numpy as np
import imageio


## visualize predictions and gt
def visualize_output(var_map):
    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_output.png'.format(kk)
        imageio.imsave(save_path + name, pred_edge_kk)


def visualize_gt(var_map):
    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk, :, :, :]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        imageio.imsave(save_path + name, pred_edge_kk)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).sum()


def train(model, opt, crit, train_loader, epoch):
    model.train()
    for i, (X, Y) in enumerate(train_loader):
        print(f'Step [{i}/{len(train_loader)}]')
        X = X.to(device)
        Y = Y.to(device)

        output = torch.sigmoid(model(X))  # .sigmoid()

        visualize_gt(Y)
        visualize_output(output)

        struct_loss = structure_loss(output, Y).to(device)

        loss = crit(output, Y) + struct_loss


        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()


def main():
    datasets = ['SIP', 'DUT-RGBD', 'NLPR', 'NJU2K']
    current_dataset = datasets[0]

    print('__Number of CUDA Devices:', cuda.device_count(), ', active:', cuda.current_device())
    print('Device name: .... ', cuda.get_device_name(cuda.current_device()), ', available >', cuda.is_available())

    # model = BaseNetwork_3.DenseNetBackbone()
    model = SODModel.GATSegmentationModel(training=True)

    model = nn.DataParallel(model)

    cudnn.benchmark = True
    model.to(device)
    # print(count_parameters(model))

    base_lr = 0.0001
    epochs = 10
    weight_decay = 1e-3
    k = 0
    total_loss = []

    optimizerr = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    # criterion = nn.SmoothL1Loss().to(device) #
    criterion = nn.MSELoss().to(device)
    # criterion = CriterionDSN().to(device)
    dataset_path = r'/home/hussaint/SODDatasets/' + current_dataset
    d_type = ['Train', 'Test']

    train_data = DatasetLoader(dataset_path, d_type[0])
    train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=16, drop_last=True)

    print('Training...')

    for epoch in range(0, epochs):

        current_loss = train(model, optimizerr, criterion, train_loader, epoch)

        print("Epoch: %d of %d, loss: %f" % (epoch, epochs, current_loss))
        total_loss.append(current_loss)

        if epoch % 25 == 24 or epoch == epochs:
            print('Saving model and weights...')
            torch.save(model, 'DDNet' + current_dataset + '_' + str(epoch) + '.pt')
            torch.save(model.state_dict(), 'DDNetWts_' + current_dataset + '_' + str(epoch) + '.pt')


if __name__ == '__main__':
    main()
