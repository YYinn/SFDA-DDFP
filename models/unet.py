from numpy import log
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeaturesSegmenter(nn.Module):

    def __init__(self, in_channels=64, out_channels=2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x_):
        x = F.relu(self.conv1(x_))
        x = F.relu(self.conv2(x))
        out = self.conv3(x)

        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, last_relu=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if last_relu:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, last_relu=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, last_relu=last_relu)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


# Eq. (1): style loss between noise image and BN statistics of source model
class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    https://github.com/CityU-AIM-Group/SFDA-FSM/blob/main/tools/domain_inversion.py
    '''
    def __init__(self, module, bn_stats=None, name=None):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.bn_stats = bn_stats
        self.name = name
        self.r_feature = None
        self.mean = None
        self.var = None

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        # r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
        #     module.running_mean.data - mean, 2)
        if self.bn_stats is None:
            var_feature = torch.norm(module.running_var.data - var, 2)
            mean_feature = torch.norm(module.running_mean.data - mean, 2)
        else:
            var_feature = torch.norm(
                torch.tensor(
                    self.bn_stats[self.name + ".running_var"], device=input[0].device
                )
                - var,
                2,
            )
            mean_feature = torch.norm(
                torch.tensor(
                    self.bn_stats[self.name + ".running_mean"], device=input[0].device
                )
                - mean,
                2,
            )
        rescale = 1.0
        self.r_feature = mean_feature + rescale * var_feature
        self.mean = mean
        self.var = var
        # must have no output

    def close(self):
        self.hook.remove()

class UNet(nn.Module):
    def __init__(self, opt, n_channels, n_classes,only_feature=True,bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.only_feature = only_feature
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear, last_relu=False)
        if self.only_feature == False:
            self.outc = OutConv(64, n_classes)

        # if opt['config_file'] not in ('configs/train_source_seg.yaml', 'configs/test_source_seg.yaml'):
        if not (opt['config_file'].startswith('configs/train_source_seg') or opt['config_file'].startswith('configs/test_source_seg')):
            self.src_params = torch.load(opt['source_model_path'], map_location='cpu')['model']

            self.bn_loss_hooks = []
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.BatchNorm2d) and not name.startswith('freconv'):
                    self.bn_loss_hooks.append(name)
            print(f'prompt layer : {self.bn_loss_hooks}')    

    def forward(self, x, only_feature = False,type=None, phase=None):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.only_feature:
            return x
        else:
            return F.relu(x),self.outc(F.relu(x)), torch.tensor(1), torch.tensor(1)
        
    def get_BNLoss(self):
        loss_bn_tmp = 0
        for i, hook in enumerate(self.bn_loss_hooks):
            loss_bn_tmp += hook.r_feature
            hook.close()
        return loss_bn_tmp, 1, 1
    
    def set_hook(self):
        self.bn_loss_hooks = []
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d) and not name.startswith('freconv'):
                self.bn_loss_hooks.append(DeepInversionFeatureHook(module, bn_stats=self.src_params, name=name))     

