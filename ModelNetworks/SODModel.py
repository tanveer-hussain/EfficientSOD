import torch
from torch import nn
import torchvision.models as models
import timm
from ExternalModules import MHSA

import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

BatchNorm2d_class = BatchNorm2d = torch.nn.SyncBatchNorm
relu_inplace = True
BN_MOMENTUM = 0.1
ALIGN_CORNERS = None
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=relu_inplace)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

class SpatialGCN(nn.Module):
    def __init__(self, plane):
        super(SpatialGCN, self).__init__()
        inter_plane = plane // 2
        self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1)
        self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1)

        self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(inter_plane)
        self.softmax = nn.Softmax(dim=2)

        self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
                                 BatchNorm2d(plane))

    def forward(self, x):
        # b, c, h, w = x.size()
        node_k = self.node_k(x)
        node_v = self.node_v(x)
        node_q = self.node_q(x)
        b,c,h,w = node_k.size()
        node_k = node_k.view(b, c, -1).permute(0, 2, 1)
        node_q = node_q.view(b, c, -1)
        node_v = node_v.view(b, c, -1).permute(0, 2, 1)
        # A = k * q
        # AV = k * q * v
        # AVW = k *(q *v) * w
        AV = torch.bmm(node_q,node_v)
        AV = self.softmax(AV)
        AV = torch.bmm(node_k, AV)
        AV = AV.transpose(1, 2).contiguous()
        AVW = self.conv_wg(AV)
        AVW = self.bn_wg(AVW)
        AVW = AVW.view(b, c, h, -1)
        out = F.relu_(self.out(AVW) + x)
        return out


class DualGCN(nn.Module):
    """
        Feature GCN with coordinate GCN
    """
    def __init__(self, planes, ratio=4):
        super(DualGCN, self).__init__()

        self.phi = nn.Conv2d(planes, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_phi = BatchNorm2d(planes // ratio * 2)
        self.theta = nn.Conv2d(planes, planes // ratio, kernel_size=1, bias=False)
        self.bn_theta = BatchNorm2d(planes // ratio)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.conv_adj = nn.Conv1d(planes // ratio, planes // ratio, kernel_size=1, bias=False)
        self.bn_adj = BatchNorm1d(planes // ratio)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(planes // ratio * 2, planes // ratio * 2, kernel_size=1, bias=False)
        self.bn_wg = BatchNorm1d(planes // ratio * 2)

        #  last fc
        self.conv3 = nn.Conv2d(planes // ratio * 2, planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes)

        self.local = nn.Sequential(
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes),
            nn.Conv2d(planes, planes, 3, groups=planes, stride=2, padding=1, bias=False),
            BatchNorm2d(planes))
        self.gcn_local_attention = SpatialGCN(planes)

        self.final = nn.Sequential(nn.Conv2d(planes * 2, planes, kernel_size=1, bias=False),
                                   BatchNorm2d(planes))

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, feat):
        # # # # Local # # # #
        x = feat
        local = self.local(feat)
        local = self.gcn_local_attention(local)
        local = F.interpolate(local, size=x.size()[2:], mode='bilinear', align_corners=True)
        spatial_local_feat = x * local + x

        # # # # Projection Space # # # #
        x_sqz, b = x, x

        x_sqz = self.phi(x_sqz)
        x_sqz = self.bn_phi(x_sqz)
        x_sqz = self.to_matrix(x_sqz)

        b = self.theta(b)
        b = self.bn_theta(b)
        b = self.to_matrix(b)

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()

        z = self.conv_adj(z)
        z = self.bn_adj(z)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        # cat or sum, nearly the same results
        out = self.final(torch.cat((spatial_local_feat, g_out), 1))

        return out


class DualGCNHead(nn.Module):
    def __init__(self, inplanes, interplanes, num_classes):
        super(DualGCNHead, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))
        self.dualgcn = DualGCN(interplanes)
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   BatchNorm2d(interplanes),
                                   nn.ReLU(interplanes))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes + interplanes, interplanes, kernel_size=3, padding=1, dilation=1, bias=False),
            BatchNorm2d(interplanes),
            nn.ReLU(interplanes),
            nn.Conv2d(interplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        output = self.conva(x)
        output = self.dualgcn(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output

from torchvision.models import resnet101, ResNet101_Weights

class Backbone(nn.Module):

    def __init__(self, output_layer):
        super(Backbone, self).__init__()
        # self.backbonemodel = timm.create_model('hrnet_w18', pretrained=True, features_only=True)
        # self.backbonemodel.eval()
        self.weights = ResNet101_Weights.DEFAULT
        self.preprocess = self.weights.transforms()
        self.backbone = resnet101(weights = self.weights)
        self.backbone.eval()

        self.children_list = []
        for n,c in self.backbone.named_children():
            self.children_list.append(c)
            if n == output_layer:
                break

        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None

        # modules = list(self.backbone.children())[:-1]
        # self.backbone = nn.Sequential(*modules)
        
        # children_counter = 0
        # for n, c in self.backbone.named_children():
        #     print("Children Counter: ",children_counter," Layer Name: ",n,)


    def forward(self, x):
        
        with torch.no_grad():
            # y = self.preprocess(x)
            y = self.net(x)
            # y = self.backbone(x)
            # y = self.backbonemodel(x)

        return y


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels,eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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

    def forward(self, q, k, v):
        n_batch, C, width, height = q.size()
        q = self.query(q).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(k).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(v).view(n_batch, self.heads, C // self.heads, -1)

        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)

        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)

        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)

        return out

from ResNet import B2_ResNet
class DDNet(nn.Module):
    
    def __init__(self, channels=64, training=False):
        super(DDNet, self).__init__()

        self.training = training

        self.resnet = B2_ResNet()
        if self.training:
            self.initialize_weights()

        self.resblock1 = BasicBlock(channels, channels)
        self.conv0 = DoubleConv(channels * 2, channels)
        self.conv1 = DoubleConv(channels * 4, channels)
        self.conv2 = DoubleConv(channels * 8, channels)
        self.conv3 = DoubleConv(channels * 16, channels)
        self.conv4 = DoubleConv(channels * 32, channels)

        self.resblock = BasicBlock(channels, channels)

        # # # DualGCN
        self.head = DualGCNHead(channels, channels, channels)

        self.convpred = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(channels // 2),
            nn.Dropout2d(0.1),
            nn.Conv2d(channels // 2, 1, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.a1 = MHSA(channels, 16, 16)
        self.a2 = MHSA(channels, 32, 32)
        # self.a3 = MHSA(channels, 64, 64)


    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        f1 = self.resnet.layer1(x)  # 256 x 64 x 64
        f2 = self.resnet.layer2(f1)  # 512 x 32 x 32
        f3 = self.resnet.layer3(f2) # 1024 x 16 x 16
        f4 = self.resnet.layer4(f3)  # 2048 x 8 x 8

        f1 = self.head(self.conv1(f1))
        f2 = self.head(self.conv2(f2))
        f3 = self.head(self.conv3(f3))
        f4 = self.head(self.conv4(f4))

        f4 = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=True)
        f43 = self.a1(self.resblock(self.conv0(torch.cat((f4, f3), 1))), f4, f3)

        f43 = F.interpolate(f43, scale_factor=2, mode='bilinear', align_corners=True)
        f432 = self.a2(self.resblock(self.conv0(torch.cat((f43, f2), 1))), f43, f2)

        f432 = F.interpolate(f432, scale_factor=2, mode='bilinear', align_corners=True)
        # f4321 = self.a3(self.conv0(torch.cat((f432, f1), 1)), f432, f1)
        f4321 = self.resblock(self.conv0(torch.cat((f432, f1), 1)))

        f4321 = F.interpolate(f4321, scale_factor=4, mode='bilinear', align_corners=True)

        f4321 = self.convpred(f4321)

        return f4321
    
    def initialize_weights(self):
        print ('Loading weights...')
        res50 = models.resnet50(pretrained=True)
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
        print (self.resnet.load_state_dict(all_params))



# from ptflops import get_model_complexity_info

# with torch.cuda.device(0):
#   net = DDNet()
#   x = torch.randn((3, 224, 224))
#   macs, params = get_model_complexity_info(net,(3, 256, 256), as_strings=True,
#                                            print_per_layer_stat=True, verbose=True)
#   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#   print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# model = DDNet(training=False)
# tensor = torch.randn((2,3,256,256))
# with torch.no_grad():
#     out = model(tensor)
#     print (out.shape)
# # for o inf
# # print (out[0].shape, out[1].shape)

# print ('Done')