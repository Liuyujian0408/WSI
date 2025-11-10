import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from model.SwinTransformer import SwinB, SwinT
from model.hvae.HVAE import HierarchicalVAE, hvae_cfg
import os
import lpips
import piq


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def make_cbr(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                         nn.BatchNorm2d(out_dim),
                         nn.PReLU())


def make_cbr_in(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                         nn.InstanceNorm2d(out_dim),
                         nn.PReLU())


def make_convtrans_cbr(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(out_dim),
        nn.PReLU()
    )


def make_cbg(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                         nn.BatchNorm2d(out_dim),
                         nn.GELU())


def make_cbg_in(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                         nn.InstanceNorm2d(out_dim),
                         nn.GELU())

def rescale_to(x, scale_factor: float = 2, interpolation='nearest'):
    return torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode=interpolation)


def resize_as(x, y, interpolation='bilinear'):
    return torch.nn.functional.interpolate(x, size=y.shape[-2:], mode=interpolation)

def image2patches(x):
    x = rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
    return x


def patches2image(x):
    x = rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x

class Upsample(nn.Sequential):


    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


# model for single-scale training
class MVANet_plus(nn.Module):
    def __init__(self, hvae_cfg, hvae_path, device=torch.device('cuda')):
        super().__init__()

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.img_range = 1

        HVAE = HierarchicalVAE(hvae_cfg)
        HVAE_dict = torch.load(hvae_path)['model']
        HVAE.load_state_dict(HVAE_dict, strict=True)
        HVAE.compress_mode()
        self.HVAE = HVAE.to(device=device).eval()
        self._256map128 = make_cbr_in(256, 128)
        self._384map128 = make_cbr_in(384, 128)

        self.conv_up1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


        self.backbone = SwinB(pretrained=True)

        self.upsample = Upsample(2, 128)

        self.sideout_512 = nn.Sequential(nn.Conv2d(128, 3, kernel_size=3, padding=1))
        self.sideout_1024 = nn.Sequential(nn.Conv2d(128, 3, kernel_size=3, padding=1))
        self.sideout_256 = nn.Sequential(nn.Conv2d(128, 3, kernel_size=3, padding=1))

        emb_dim = 128
        self.insmask_head = nn.Sequential(
            nn.Conv2d(emb_dim, 384, kernel_size=3, padding=1),
            nn.InstanceNorm2d(384),
            nn.PReLU(),
            nn.Conv2d(384, emb_dim, kernel_size=3, padding=1)
        )

        self.upsample1 = make_cbg_in(emb_dim, emb_dim)
        self.upsample2 = make_cbg_in(emb_dim, emb_dim)
        self.output = nn.Sequential(nn.Conv2d(emb_dim, 3, kernel_size=3, padding=1))


    def forward(self, x, name):

        loc = image2patches(x) 

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        glb = rescale_to(x, scale_factor=0.5, interpolation='bilinear') 

        glb_feature = self.backbone(glb)[2] 
        assert hasattr(self.HVAE, 'decompress_file')

        tmp_bits_path = f'temp/{name}.bits'

        with torch.no_grad():
            if not os.path.exists(tmp_bits_path):
                os.makedirs(os.path.dirname(tmp_bits_path), exist_ok=True)
                self.HVAE.compress_file(loc, tmp_bits_path)
            fake, features_list = self.HVAE.decompress_file(tmp_bits_path)
        features_list = features_list[1]

        features = self._256map128(glb_feature)

        local = self._384map128(patches2image(features_list)) 

        final_output_128 = features + local  
        final_output_256 = self.lrelu(
            self.conv_up1(torch.nn.functional.interpolate(final_output_128, scale_factor=2, mode='nearest')))
        final_output_512 = self.lrelu(self.conv_up2(
            torch.nn.functional.interpolate(final_output_256, scale_factor=2, mode='nearest')))

        final_output_512 = self.insmask_head(final_output_512) 
        final_output_512 = final_output_512 
        final_output_1024 = self.upsample1(rescale_to(final_output_512)) 
        final_output = rescale_to(final_output_1024) 

       
        final_output = self.upsample2(final_output)
        final_output = self.output(final_output)


        sideout_256 = self.sideout_256(final_output_256) 
        sideout_512 = self.sideout_512(final_output_512) 
        sideout_1024 = self.sideout_1024(final_output_1024)  

        if self.training:
            return sideout_256, sideout_512, sideout_1024, final_output
        else:
            return final_output

