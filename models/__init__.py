from models.unet import UNet
from models.DeepLabV3Plus.network import deeplabv3plus_resnet50
from models.prompt_unet import Pmt_UNet
from models.deeplabv3 import Deeplabv3plus_res50
from models.pmt_deeplabv3 import pmt_Deeplabv3plus_res50
# from monai.networks.nets import SwinUNETR
from models.swinunetr import SwinUNETR

def get_model(cfg):
    
    if cfg['arch'] == 'UNet':
        model = UNet(cfg, n_channels=cfg['input_dim'],n_classes=cfg['num_classes'],only_feature=False)
    elif cfg['arch'] == 'DeepLab':
        # model = deeplabv3plus_resnet50(num_classes=cfg['num_classes'],only_feature=False)
        model = Deeplabv3plus_res50(cfg, num_classes=cfg['num_classes'])
    elif cfg['arch'] == 'Pmt_DeepLab':
        # model = deeplabv3plus_resnet50(num_classes=cfg['num_classes'],only_feature=False)
        model = pmt_Deeplabv3plus_res50(cfg, num_classes=cfg['num_classes'])
    elif cfg['arch'] == 'Pmt_UNet':
        model = Pmt_UNet(cfg, n_channels=cfg['input_dim'],n_classes=cfg['num_classes'],only_feature=False)
    elif cfg['arch'] == 'SwinUNETR':
        model = SwinUNETR(cfg, img_size=(256, 256), in_channels=3, out_channels=cfg['num_classes'], use_checkpoint=True, spatial_dims=2)
    return model