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
        self.cnt = 0

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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Pmt_UNet(nn.Module):
    def __init__(self, opt, n_channels, n_classes,only_feature=True,bilinear=False):
        super(Pmt_UNet, self).__init__()

        ## unet setting
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

        ## prompt setting
        self.pmt_size = opt['pmt_size']
        self.prompt_a = nn.Parameter(torch.zeros(1, 1, self.pmt_size, self.pmt_size), requires_grad=True) ## trainable domain-dependent prompt
        self.freconv_c = 3
        self.pmtinc = 3 ## if ablation only with amp, change pmtinc to 2
        self.frefuse_c = 4

        if opt['pmt_type'] == 'Data':
            self.freconv_a = nn.Sequential(
                nn.Conv2d(1, self.freconv_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.freconv_c, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.freconv_c, 1, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(1, momentum=0.1),
                nn.ReLU(inplace=True),
            )
            self.freconv_p = nn.Sequential(
                nn.Conv2d(1, self.freconv_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.freconv_c, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.freconv_c, 1, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(1, momentum=0.1),
                nn.ReLU(inplace=True),
            )
            self.freconv = nn.Sequential(
                nn.Conv2d(self.pmtinc, self.frefuse_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.frefuse_c, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.frefuse_c, self.frefuse_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.frefuse_c, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.frefuse_c, self.pmtinc, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(self.pmtinc, momentum=0.1),
                nn.ReLU(inplace=True)
            )

        ### aligning target model's bn statistic with which model (default: source, you can also switch the opt['bn_align_model'] ot other model(only for ablation))
        if opt['doing'] == 'train':
            self.src_params = torch.load(opt['bn_align_model'], map_location='cpu')['model']
        else:
            self.src_params = torch.load(opt['source_model_path'], map_location='cpu')['model'] # not going to use for any calculation

        self.bn_loss_hooks = []
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d) and not name.startswith('freconv'):
                self.bn_loss_hooks.append(name)
        print(f'prompt layer : {self.bn_loss_hooks}')   

        self.alpha = opt['alpha']  
        

    def Data_singlea(self, ori_a, ori_p=None, ori_pmt=None):
        '''
        Use only amptitude and prompt for data-specific frequency prompt generation
        [Only used for ablation]
        '''
        conv_a = self.freconv_a(ori_a)
        pmt = ori_pmt.repeat(ori_a.shape[0], 1, 1, 1)   # [1, 1, h, w] -> [B, 1, h, w]
        data_specific_prompt_a = self.freconv(torch.stack([conv_a, pmt], dim=1).squeeze(2))[:, 1:, ...]#in [16, 2, 256, 256] out [16, 1, 256, 256]
        data_specific_prompt_a = data_specific_prompt_a * (1-self.alpha) + ori_pmt * self.alpha
        return data_specific_prompt_a
    def Data_singlep(self, ori_a, ori_p=None, ori_pmt=None):
        '''
        Use only phase and prompt for data-specific frequency prompt generation
        [Only used for ablation]
        '''
        conv_p = self.freconv_p(ori_p)
        pmt = ori_pmt.repeat(ori_a.shape[0], 1, 1, 1)
        data_specific_prompt_a = self.freconv(torch.stack([conv_p, pmt], dim=1).squeeze(2))[:, 1:, ...]#in [16, 2, 256, 256] out [16, 1, 256, 256]
        data_specific_prompt_a = data_specific_prompt_a * (1-self.alpha) + ori_pmt * self.alpha
        return data_specific_prompt_a

    def Data_multi(self, ori_a, ori_p=None, ori_pmt=None):
        '''
        Use amplitude, phase and prompt for data-specific frequency prompt generation
        '''
        conv_a = self.freconv_a(ori_a)
        conv_p = self.freconv_p(ori_p)
        pmt = ori_pmt.repeat(ori_a.shape[0], 1, 1, 1)
        data_specific_prompt_a = self.freconv(torch.stack([conv_a, conv_p, pmt], dim=1).squeeze(2))[:, 2:, ...]#in [16, 3, 256, 256] out [16, 1, 256, 256]
        data_specific_prompt_a = data_specific_prompt_a * (1-self.alpha) + ori_pmt * self.alpha
        return data_specific_prompt_a
    
    def Data_multi_pad4(self, ori_a, ori_p=None, ori_pmt=None, p=0):
        '''
        Use amplitude, phase and prompt for data-specific frequency prompt generation
        When the prompt size is smaller than the image, we need to pad it
        '''
        assert ori_pmt.shape[2] + p * 2 == 256
        conv_a = self.freconv_a(ori_a[:, :, p: p+ori_pmt.shape[2], p: p+ori_pmt.shape[2]])
        conv_p = self.freconv_p(ori_p[:, :, p: p+ori_pmt.shape[2], p: p+ori_pmt.shape[2]])
        pmt = ori_pmt.repeat(ori_a.shape[0], 1, 1, 1)
        data_specific_prompt_a = self.freconv(torch.stack([conv_a, conv_p, pmt], dim=1).squeeze(2))[:, 2:, ...]#in [16, 3, 256, 256] out [16, 1, 256, 256]
        data_specific_prompt_a = data_specific_prompt_a + pmt
        pad_func = nn.ConstantPad2d(padding=(p, p, p, p), value=1)
        data_specific_prompt_a = pad_func(data_specific_prompt_a)        
        return data_specific_prompt_a
    
    def Domain_multi(self, ori_a, ori_p=None, ori_pmt=None):
        '''
        Domain-specific frequency prompt generation
        '''
        pmt = ori_pmt.repeat(ori_a.shape[0], 1, 1, 1)
        return pmt
    
    def Domain_multi_pad4(self, ori_a, ori_p=None, ori_pmt=None, padding_shape=0):
        '''
        Domain-specific frequency prompt generation (padding scene).
        '''
        pmt = ori_pmt.repeat(ori_a.shape[0], 1, 1, 1)
        pad_func = nn.ConstantPad2d(padding=(padding_shape, padding_shape, padding_shape, padding_shape), value=1)
        pmt = pad_func(pmt)
        return pmt

    def fre_prompt(self, input, mask=None, type='Domain', phase=None):
        '''
        1. Since the image is 3 channel (some augmentation processed, so channels are not the same), 
        we used the prompt(1 channel) for each image channel(can be seen as a augmenation process)

        2. why exp: to make sure the prompt > 0 (fourier space characteristic). See more details in paper. 
        '''
        new_img = []
        for i in range(3): 
            # i = 1 ## used for ablation
            fft_np = torch.fft.fft2(input[:, i:i+1, ...], dim=(-2, -1))
            amp, pha = torch.abs(fft_np), torch.angle(fft_np)

            a = torch.fft.fftshift(amp, dim=(-2, -1))
            p = torch.fft.fftshift(pha, dim=(-2, -1))

            if type == 'Domain':
                if self.pmt_size < 256:
                    paddingsize = int((256 - self.pmt_size) / 2)
                    prompt = self.Domain_multi_pad4(a, p, torch.exp(self.prompt_a), paddingsize)
                else:
                    prompt = self.Domain_multi(a, p, torch.exp(self.prompt_a))
                    ## used for ablation
                    # prompt = self.Domain_multi(a, p, self.prompt_a)
            elif type == 'Data':
                if self.pmt_size < 256:
                    paddingsize = int((256 - self.pmt_size) / 2)
                    prompt = self.Data_multi_pad4(a, p, torch.exp(self.prompt_a), paddingsize)
                else:
                    prompt = self.Data_multi(a, p, torch.exp(self.prompt_a))
                    ## used for ablation
                    # prompt = self.Data_singlea(a, p, torch.exp(self.prompt_a))
                    # prompt = self.Data_multi(a, p, self.prompt_a)
            else:
                raise ValueError('Please input Domain or Data as type')
            
            a = prompt * a

            a = torch.fft.ifftshift(a, dim=(-2, -1))

            fft_ = a * torch.exp( 1j * pha )

            prompted_img = torch.fft.ifft2(fft_, dim=(-2, -1))
            prompted_img = torch.real(prompted_img)
        
            new_img.append(prompted_img)

        new_img = torch.stack((new_img[0], new_img[1], new_img[2]), dim=1).squeeze(2)
        ## used for ablation
        # new_img = torch.stack((prompted_img, prompted_img, prompted_img), dim=1).squeeze(2)
       
        return new_img, prompt

    def set_hook(self):
        bn_cnt = 0
        self.bn_loss_hooks = []
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d) and not name.startswith('freconv'):
                self.bn_loss_hooks.append(DeepInversionFeatureHook(module, bn_stats=self.src_params, name=name))     
                bn_cnt += 1

    def spa_prompt(self, input):
        return input + self.prompt_a, self.prompt_a

    def forward(self, x, only_feature = False, type=None, phase=None):
        if type == 'None' or type == None:
            prompted_img = x
            prompt_a = torch.zeros_like(x).cuda()
        elif type == 'Spatial':
            prompted_img, prompt_a = self.spa_prompt(x)
        elif type == 'Domain' or type == 'Data':
            prompted_img, prompt_a = self.fre_prompt(x, type=type, phase=phase)
        else:
            raise ValueError(f'Please input Domain | Data | spacial as type, but got {type} instead ')

        x1 = self.inc(prompted_img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return x,self.outc(F.relu(x)), prompted_img, prompt_a
        
    def get_BNLoss(self):
        '''
        calculate BN loss
        '''
        loss_bn_tmp = 0
        bn_mean = []
        bn_var = []
        for i, hook in enumerate(self.bn_loss_hooks):
            loss_bn_tmp += hook.r_feature 
            bn_mean.append(hook.mean.detach().cpu().numpy())
            bn_var.append(hook.var.detach().cpu().numpy())
            hook.close()
        return loss_bn_tmp, bn_mean, bn_var

