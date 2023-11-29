from torch import nn, cuda
import torch
import torch.nn.functional as F
import numpy as np
from scipy import misc
# ################ UCNet model code #########################

## visualize predictions and gt
def visualize_output(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_output.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)

def visualize_gt(var_map):

    for kk in range(var_map.shape[0]):
        pred_edge_kk = var_map[kk,:,:,:]
        pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
        # pred_edge_kk = (pred_edge_kk - pred_edge_kk.min()) / (pred_edge_kk.max() - pred_edge_kk.min() + 1e-8)
        pred_edge_kk *= 255.0
        pred_edge_kk = pred_edge_kk.astype(np.uint8)
        save_path = './temp/'
        name = '{:02d}_gt.png'.format(kk)
        misc.imsave(save_path + name, pred_edge_kk)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        self.asppconv = torch.nn.Sequential()
        if bn_start:
            self.asppconv = nn.Sequential(
                nn.BatchNorm2d(input_num),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        else:
            self.asppconv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        self.drop_rate = drop_out

    def forward(self, _input):
        #feature = super(_DenseAsppBlock, self).forward(_input)
        feature = self.asppconv(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class multi_scale_aspp(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, channel):
        super(multi_scale_aspp, self).__init__()
        self.ASPP_3 = _DenseAsppBlock(input_num=channel, num1=channel * 2, num2=channel, dilation_rate=3,
                                      drop_out=0.1, bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(input_num=channel * 2, num1=channel * 2, num2=channel,
                                      dilation_rate=6, drop_out=0.1, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=channel * 3, num1=channel * 2, num2=channel,
                                       dilation_rate=12, drop_out=0.1, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=channel * 4, num1=channel * 2, num2=channel,
                                       dilation_rate=18, drop_out=0.1, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=channel * 5, num1=channel * 2, num2=channel,
                                       dilation_rate=24, drop_out=0.1, bn_start=True)
        self.classification = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=channel * 6, out_channels=channel, kernel_size=1, padding=0)
        )

    def forward(self, _input):
        #feature = super(_DenseAsppBlock, self).forward(_input)
        aspp3 = self.ASPP_3(_input)
        feature = torch.cat((aspp3, _input), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)

        feature = torch.cat((aspp24, feature), dim=1)

        aspp_feat = self.classification(feature)

        return aspp_feat

class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        self.asppconv = torch.nn.Sequential()
        if bn_start:
            self.asppconv = nn.Sequential(
                nn.BatchNorm2d(input_num),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        else:
            self.asppconv = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1),
                nn.BatchNorm2d(num1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                          dilation=dilation_rate, padding=dilation_rate)
            )
        self.drop_rate = drop_out

    def forward(self, _input):
        #feature = super(_DenseAsppBlock, self).forward(_input)
        feature = self.asppconv(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Triple_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Triple_Conv, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            # BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)

class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self):
        super(CAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)

class RCAB(nn.Module):
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res
###############################################################
###################### ######################
# https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MHSA(nn.Module):
    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads

        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads, width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

###############################################################
###################### ######################
# https://github.com/yjn870/DRRN-pytorch/blob/master/models.py

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvLayer, self).__init__()
        self.module = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
        )

    def forward(self, x):
        return self.module(x)


class ResidualUnit(nn.Module):
    def __init__(self, num_features):
        super(ResidualUnit, self).__init__()
        self.module = nn.Sequential(
            ConvLayer(num_features, num_features),
            ConvLayer(num_features, num_features)
        )

    def forward(self, h0, x):
        return h0 + self.module(x)


class RecursiveBlock(nn.Module):
    def __init__(self, in_channels, out_channels, U):
        super(RecursiveBlock, self).__init__()
        self.U = U
        self.h0 = ConvLayer(in_channels, out_channels)
        self.ru = ResidualUnit(out_channels)

    def forward(self, x):
        h0 = self.h0(x)
        x = h0
        for i in range(self.U):
            x = self.ru(h0, x)
        return x


###############################################################
# https://github.com/JingZhang617/UCNet/blob/4d595997a71bca93aa90df2dc46199cdf1b20e7c/model/ResNet_models.py#L296
###################### ######################
class multi_scale_aspp(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, channel):
        super(multi_scale_aspp, self).__init__()
        self.ASPP_3 = _DenseAsppBlock(input_num=channel, num1=channel * 2, num2=channel, dilation_rate=3,
                                      drop_out=0.1, bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(input_num=channel * 2, num1=channel * 2, num2=channel,
                                      dilation_rate=6, drop_out=0.1, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=channel * 3, num1=channel * 2, num2=channel,
                                       dilation_rate=12, drop_out=0.1, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=channel * 4, num1=channel * 2, num2=channel,
                                       dilation_rate=18, drop_out=0.1, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=channel * 5, num1=channel * 2, num2=channel,
                                       dilation_rate=24, drop_out=0.1, bn_start=True)
        self.classification = nn.Sequential(
            nn.Dropout2d(p=0.1),
            nn.Conv2d(in_channels=channel * 6, out_channels=channel, kernel_size=1, padding=0)
        )

    def forward(self, _input):
        #feature = super(_DenseAsppBlock, self).forward(_input)
        aspp3 = self.ASPP_3(_input)
        feature = torch.cat((aspp3, _input), dim=1)

        aspp6 = self.ASPP_6(feature)
        feature = torch.cat((aspp6, feature), dim=1)

        aspp12 = self.ASPP_12(feature)
        feature = torch.cat((aspp12, feature), dim=1)

        aspp18 = self.ASPP_18(feature)
        feature = torch.cat((aspp18, feature), dim=1)

        aspp24 = self.ASPP_24(feature)

        feature = torch.cat((aspp24, feature), dim=1)

        aspp_feat = self.classification(feature)

        return aspp_feat