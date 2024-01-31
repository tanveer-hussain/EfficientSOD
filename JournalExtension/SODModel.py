import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from ResNet import B2_ResNet
from torch.autograd import Variable
from torch.nn import Parameter, Softmax
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=5):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Encoder_x(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_x, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2 = nn.Linear(channels * 8 * 11 * 11, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, input):
        output = self.leakyrelu(self.bn1(self.layer1(input)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))
        output = output.view(-1, self.channel * 8 * 11 * 11)
        # print(output.size())
        # output = self.tanh(output)

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        # print(output.size())
        # output = self.tanh(output)

        return dist, mu, logvar

class Encoder_xy(nn.Module):
    def __init__(self, input_channels, channels, latent_size):
        super(Encoder_xy, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Conv2d(input_channels, channels, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.layer2 = nn.Conv2d(channels, 2*channels, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channels * 2)
        self.layer3 = nn.Conv2d(2*channels, 4*channels, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.layer4 = nn.Conv2d(4*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channels * 8)
        self.layer5 = nn.Conv2d(8*channels, 8*channels, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channels * 8)
        self.channel = channels

        self.fc1 = nn.Linear(channels * 8 * 11 * 11, latent_size)
        self.fc2 = nn.Linear(channels * 8 * 11 * 11, latent_size)

        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        output = self.leakyrelu(self.bn1(self.layer1(x)))
        # print(output.size())
        output = self.leakyrelu(self.bn2(self.layer2(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn3(self.layer3(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer4(output)))
        # print(output.size())
        output = self.leakyrelu(self.bn4(self.layer5(output)))
        output = output.view(-1, self.channel * 8 * 11 * 11)

        mu = self.fc1(output)
        logvar = self.fc2(output)
        dist = Independent(Normal(loc=mu, scale=torch.exp(logvar)), 1)
        # print(output.size())
        # output = self.tanh(output)

        return dist, mu, logvar

class Generator(nn.Module):
    def __init__(self, channel, latent_dim):
        super(Generator, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sal_encoder = Saliency_feat_encoder(channel, latent_dim)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.xy_encoder = Encoder_xy(7, channel, latent_dim)
        self.x_encoder = Encoder_x(6, channel, latent_dim)
        self.tanh = nn.Tanh()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def kl_divergence(self, posterior_latent_space, prior_latent_space):
        kl_div = kl.kl_divergence(posterior_latent_space, prior_latent_space)
        return kl_div

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x, depth, y=None, training=True):
        if training:
            self.posterior, muxy, logvarxy = self.xy_encoder(torch.cat((x,depth,y),1))
            self.prior, mux, logvarx = self.x_encoder(torch.cat((x,depth),1))
            lattent_loss = torch.mean(self.kl_divergence(self.posterior, self.prior))
            z_noise_post = self.reparametrize(muxy, logvarxy)
            z_noise_prior = self.reparametrize(mux, logvarx)
            self.prob_pred_post, self.depth_pred_post  = self.sal_encoder(x,depth,z_noise_post)
            self.prob_pred_prior, self.depth_pred_prior = self.sal_encoder(x, depth, z_noise_prior)
            return self.prob_pred_post, self.prob_pred_prior, lattent_loss, self.depth_pred_post, self.depth_pred_prior
        else:
            _, mux, logvarx = self.x_encoder(torch.cat((x,depth),1))
            z_noise = self.reparametrize(mux, logvarx)
            self.prob_pred,_  = self.sal_encoder(x,depth,z_noise)
            return self.prob_pred

class ChannelReducer(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio=16):
        super(ChannelReducer, self).__init__()

        # 1x1 Convolution layers to reduce channels gradually
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

        # Channel pooling to further reduce channels
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Channel attention blocks
        self.se1 = self._make_se_block(in_channels // 2, reduction_ratio)
        self.se2 = self._make_se_block(in_channels // 4, reduction_ratio)
        self.se3 = self._make_se_block(out_channels, reduction_ratio)
        self.se4 = self._make_se_block(out_channels, reduction_ratio)

    def _make_se_block(self, in_channels, reduction_ratio):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply 1x1 convolutions
        x1 = torch.relu(self.conv1(x))
        x1 = x1 * self.se1(x1)  # Apply channel attention

        x2 = torch.relu(self.conv2(x1))
        x2 = x2 * self.se2(x2)  # Apply channel attention

        x3 = torch.relu(self.conv3(x2))
        x3 = x3 * self.se3(x3)  # Apply channel attention

        # Channel pooling to reduce spatial dimensions to 1x1
        # x3 = self.pool(x3)

        return x3  # Remove singleton spatial dimensions


class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPModule, self).__init__()

        # ASPP convolutions with different dilation rates
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Apply ASPP convolutions with different dilation rates
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)

        # Global average pooling
        global_pool = nn.functional.adaptive_avg_pool2d(x, 1)
        global_pool = self.conv5(global_pool)
        global_pool = nn.functional.interpolate(global_pool, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Concatenate ASPP branches and global pooling
        out = torch.cat((out1, out2, out3, out4, global_pool), dim=1)
        return out
class WeightedFusionAttentionCNN(nn.Module):
    def __init__(self, in_channels):
        super(WeightedFusionAttentionCNN, self).__init__()

        # Upsampling layers to match the final size
        self.up = nn.Upsample(size=(64, 64), mode='bilinear')

        # Convolutional layers for each input
        self.conv2 = nn.Conv2d(in_channels * 2, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels * 2, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels * 2, 256, kernel_size=3, padding=1)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),  # Adjust channels for attention
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.final_conv = ASPPModule(768, 1)

    def forward(self, x2, x3, x4):
        # Upsample smaller inputs to match the size of the largest one
        x2 = self.up(x2)
        x3 = self.up(x3)
        x4 = self.up(x4)

        # Apply convolutional layers to each input
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        # Concatenate the feature maps
        fused = torch.cat((x2, x3, x4), dim=1)

        # Apply attention mechanism
        attention_weights = self.attention(fused)

        # Apply attention to fused features
        fused_attention = fused * attention_weights

        # Final convolutional layer for output
        output = self.final_conv(fused_attention)

        return output

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

from torch_geometric.nn import GATConv
from torch_geometric.data import Data
class Saliency_feat_encoder(nn.Module):
    def __init__(self, channel, latent_dim):
        super(Saliency_feat_encoder, self).__init__()

        self.resnet = B2_ResNet()
        self.channels = 32

        # self.training = training
        # if self.training:
        self.initialize_weights()



        self.conv2_reduce = ChannelReducer(512, 64)
        self.conv3_reduce = ChannelReducer(1024, 64)
        self.conv4_reduce = ChannelReducer(2048, 64)

        self.gatconv21 = GAT(in_channels=64, hidden_channels=64, out_channels=self.channels, heads=4)

        #
        self.gatconv31 = GATConv(in_channels=64, hidden_channels=64, out_channels=self.channels, heads=4)

        #
        self.gatconv41 = GATConv(in_channels=64, hidden_channels=64, out_channels=self.channels, heads=4)

        self.wghted_attn = WeightedFusionAttentionCNN(self.channels)

        self.up = nn.Upsample(size=(256, 256), mode='bilinear')
        self.conv_pred = nn.Conv2d(5,1,1)

        #####
        self.spatial_axes = [2, 3]
        self.xy_encoder = Encoder_xy(7, self.channels, latent_size=3)
        self.x_encoder = Encoder_x(6, self.channels, latent_size=3)
        self.conv_depth1 = BasicConv2d(6 + 3, 3, kernel_size=3, padding=1)

    def image_to_graph(self, input, radius, size):
        height, width = input.shape[-2], input.shape[-1]
        num_nodes = height * width

        # Constructing edge indices by considering connections within a radius
        edge_index = []
        for i in range(height):
            for j in range(width):
                node_idx = i * width + j
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        ni, nj = i + dx, j + dy
                        if 0 <= ni < height and 0 <= nj < width:
                            neighbor_idx = ni * width + nj
                            if node_idx != neighbor_idx:  # Exclude self-loops
                                edge_index.append((node_idx, neighbor_idx))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)

        # Ensure each edge exists in both directions (undirected graph)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        x = input.view(-1, size * size)  # Assuming the input is shaped for the ResNet
        data = Data(x=x, edge_index=edge_index)

        return data

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            device)
        return torch.index_select(a, dim, order_index)

    def forward(self, x, depth, z):

        z = torch.unsqueeze(z, 2)
        z = self.tile(z, 2, x.shape[self.spatial_axes[0]])
        z = torch.unsqueeze(z, 3)
        z = self.tile(z, 3, x.shape[self.spatial_axes[1]])
        x = torch.cat((x, depth, z), 1)
        x = self.conv_depth1(x)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4(x3)  # 2048 x 8 x 8

        # x = self.resnet.conv1(input)
        # x = self.resnet.bn1(x)
        # x = self.resnet.relu(x)
        # x = self.resnet.maxpool(x)
        #
        # x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        # x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        # x3 = self.resnet.layer3(x2)  # 1024 x 16 x 16
        # x4 = self.resnet.layer4(x3)  # 2048 x 8 x 8

        x2 = self.conv2_reduce(x2)
        x3 = self.conv3_reduce(x3)
        x4 = self.conv4_reduce(x4)

        # data2 = self.image_to_graph(x2, radius=5, size=32)  # 64x64x64 > x=[524288,1], edge_index=[2,1572864]
        # # x2, edge_index2 = data2.x, data2.edge_index
        # x2 = F.relu(self.gatconv21(x2.view(-1, 64), data2.edge_index))
        # # x2 = F.dropout(x2, p=0.5, training=self.training)
        # # x2 = F.relu(self.gatconv22(x2, edge_index2))
        # y2 = x2.view(-1, 32, 32).unsqueeze(0)
        #
        # data3 = self.image_to_graph(x3, radius=5, size=16)  # 64x64x64 > x=[524288,1], edge_index=[2,1572864]
        # # x3, edge_index3 = data3.x, data3.edge_index
        # x3 = F.relu(self.gatconv31(x3.view(-1, 64), data3.edge_index))
        # # x3 = F.dropout(x3, p=0.5, training=self.training)
        # # x3 = F.relu(self.gatconv32(x3, edge_index3))
        # y3 = x3.view(-1, 16, 16).unsqueeze(0)
        #
        # data4 = self.image_to_graph(x4, radius=5, size=8)  # 64x64x64 > x=[524288,1], edge_index=[2,1572864]
        # # x4, edge_index4 = data4.x, data4.edge_index
        # x4 = F.relu(self.gatconv41(x4.view(-1, 64), data4.edge_index))
        # # x4 = F.dropout(x4, p=0.5, training=self.training)
        # # x4 = F.relu(self.gatconv42(x4, edge_index4))
        # y4 = x4.view(-1, 8, 8).unsqueeze(0)
        print (x2.shape, x3.shape, x4.shape)

        # y = self.wghted_attn(x2, x3, x4)
        #
        # y = self.up(y)
        # y = self.conv_pred(y)


        return x4

    def initialize_weights(self):
        print('Loading weights...')
        res50 = models.resnet101(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        print(self.resnet.load_state_dict(all_params))

# from ptflops import get_model_complexity_info

# with torch.cuda.device(0):
#   net = GATSegmentationModel(training=False)
#   x = torch.randn((3, 224, 224))
#   macs, params = get_model_complexity_info(net,(3, 256, 256), as_strings=True,
#                                            print_per_layer_stat=True, verbose=True)
#   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#   print('{:<30}  {:<8}'.format('Number of parameters: ', params))


batch_size = 2
model = Generator(32,3).to(device)
tensor = torch.randn((batch_size, 3, 352, 352)).to(device)
depth = torch.randn((batch_size, 3, 352, 352)).to(device)
gt = torch.randn((batch_size, 1, 352, 352)).to(device)

with torch.no_grad():
    out = model(tensor, depth, gt)
    #print(out.shape)


print('Done')
