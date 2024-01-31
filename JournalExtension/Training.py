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
def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

## define loss
import smoothness
CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
smooth_loss = smoothness.smoothness_loss(size_average=True)


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).sum()


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




## linear annealing to avoid posterior collapse
def linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

def train(generator, generator_optimizer, crit, train_loader, epoch, epochs):
    model.train()
    for i, (images, gts, depths) in enumerate(train_loader):
        print(f'Epoch [{epoch}/{epochs}], Step [{i}/{len(train_loader)}]')
        images = images.to(device)
        gts = gts.to(device)
        depths = depths.to(device)


        pred_post, pred_prior, latent_loss, depth_pred_post, depth_pred_prior = generator.forward(images, depths, gts)

        ## l2 regularizer the inference model
        reg_loss = l2_regularisation(generator.xy_encoder) + \
                   l2_regularisation(generator.x_encoder) + l2_regularisation(generator.sal_encoder)
        smoothLoss_post = opt.sm_weight * smooth_loss(torch.sigmoid(pred_post), gts)
        reg_loss = opt.reg_weight * reg_loss
        latent_loss = latent_loss
        depth_loss_post = opt.depth_loss_weight * mse_loss(torch.sigmoid(depth_pred_post), depths)
        sal_loss = structure_loss(pred_post, gts) + smoothLoss_post + depth_loss_post
        anneal_reg = linear_annealing(0, 1, epoch, opt.epoch)
        latent_loss = opt.lat_weight * anneal_reg * latent_loss
        gen_loss_cvae = sal_loss + latent_loss
        gen_loss_cvae = opt.vae_loss_weight * gen_loss_cvae

        smoothLoss_prior = opt.sm_weight * smooth_loss(torch.sigmoid(pred_prior), gts)
        depth_loss_prior = opt.depth_loss_weight * mse_loss(torch.sigmoid(depth_pred_prior), depths)
        gen_loss_gsnn = structure_loss(pred_prior, gts) + smoothLoss_prior + depth_loss_prior
        gen_loss_gsnn = (1 - opt.vae_loss_weight) * gen_loss_gsnn
        gen_loss = gen_loss_cvae + gen_loss_gsnn + reg_loss

        generator_optimizer.zero_grad()
        gen_loss.backward()
        generator_optimizer.step()



    return gen_loss.item()


def main():
    datasets = ['SIP', 'DUT-RGBD', 'NLPR', 'NJU2K']
    current_dataset = datasets[0]

    print('__Number of CUDA Devices:', cuda.device_count(), ', active:', cuda.current_device())
    print('Device name: .... ', cuda.get_device_name(cuda.current_device()), ', available >', cuda.is_available())

    # model = BaseNetwork_3.DenseNetBackbone()
    generator = Generator(channel=feat_channel, latent_dim=latent_dim)
    generator.to(device)

    generator_params = generator.parameters()
    generator_optimizer = torch.optim.Adam(generator_params, lr_gen, betas=[beta1_gen, 0.999])

    generator = nn.DataParallel(generator)

    cudnn.benchmark = True
    # print(count_parameters(model))

    base_lr = 0.0001
    epochs = 10
    weight_decay = 1e-3
    k = 0
    total_loss = []


    print('Model on GPU: ', next(generator.parameters()).is_cuda)

    dataset_path = r'/home/hussaint/SODDatasets/' + current_dataset
    d_type = ['Train', 'Test']

    train_data = DatasetLoader(dataset_path, d_type[0])
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=16, drop_last=True)

    print('Training...')

    for epoch in range(0, epochs):

        current_loss = train(generator, generator_optimizer, None, train_loader, epoch, epochs)
        # if epoch%2 == 0:
        print("Epoch: %d of %d, loss: %f" % (epoch, epochs, current_loss))
        total_loss.append(current_loss)

        if epoch % 25 == 24 or epoch == epochs:
            print('Saving model and weights...')
            torch.save(generator, 'DDNet' + current_dataset + '_' + str(epoch) + '.pt')
            torch.save(generator.state_dict(), 'DDNetWts_' + current_dataset + '_' + str(epoch) + '.pt')


if __name__ == '__main__':
    main()
