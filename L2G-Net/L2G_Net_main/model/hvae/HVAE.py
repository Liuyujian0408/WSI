import math
import torch.nn as nn
import torch
import model.hvae.common as common
import model.hvae.vae_utils as qres
import torch.nn.functional as tnf
from collections import OrderedDict
from PIL import Image
import torchvision.transforms.functional as tvf
from torchvision.transforms.functional import to_pil_image

import pickle
import os


def hvae_cfg():
    lmb = 2048
    cfg = dict()

    enc_nums = [6, 6, 6, 4, 2]
    dec_nums = [1, 2, 3, 3, 3]
    z_dims = [16, 14, 12, 10, 8]

    im_channels = 3
    ch = 96  # 128
    cfg['enc_blocks'] = [
        common.patch_downsample(im_channels, ch * 2, rate=4),
        *[qres.MyConvNeXtBlock(ch * 2, kernel_size=7) for _ in range(enc_nums[0])],
        qres.MyConvNeXtPatchDown(ch * 2, ch * 4),
        *[qres.MyConvNeXtBlock(ch * 4, kernel_size=7) for _ in range(enc_nums[1])],
        qres.MyConvNeXtPatchDown(ch * 4, ch * 4),
        *[qres.MyConvNeXtBlock(ch * 4, kernel_size=5) for _ in range(enc_nums[2])],
        qres.MyConvNeXtPatchDown(ch * 4, ch * 4),
        *[qres.MyConvNeXtBlock(ch * 4, kernel_size=3) for _ in range(enc_nums[3])],
        qres.MyConvNeXtPatchDown(ch * 4, ch * 4),
        *[qres.MyConvNeXtBlock(ch * 4, kernel_size=1) for _ in range(enc_nums[4])],
    ]
    cfg['dec_blocks'] = [
        *[qres.QLatentBlockX(ch * 4, z_dims[0], kernel_size=1) for _ in range(dec_nums[0])],
        common.patch_upsample(ch * 4, ch * 4, rate=2),
        *[qres.QLatentBlockX(ch * 4, z_dims[1], kernel_size=3) for _ in range(dec_nums[1])],
        common.patch_upsample(ch * 4, ch * 4, rate=2),
        *[qres.QLatentBlockX(ch * 4, z_dims[2], kernel_size=5) for _ in range(dec_nums[2])],
        common.patch_upsample(ch * 4, ch * 4, rate=2),
        *[qres.QLatentBlockX(ch * 4, z_dims[3], kernel_size=7) for _ in range(dec_nums[3])],
        common.patch_upsample(ch * 4, ch * 2, rate=2),
        *[qres.QLatentBlockX(ch * 2, z_dims[4], kernel_size=7) for _ in range(dec_nums[4])],
        common.patch_upsample(ch * 2, im_channels, rate=4)
    ]
    cfg['out_net'] = qres.MSEOutputNet(mse_lmb=lmb)

    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    cfg['max_stride'] = 64
    return cfg

class BottomUpEncoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.enc_blocks = nn.ModuleList(blocks)

    def forward(self, x):
        feature = x
        enc_features = dict()
        for i, block in enumerate(self.enc_blocks):
            feature = block(feature)
            res = int(feature.shape[2])
            enc_features[res] = feature
        return enc_features

class TopDownDecoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.dec_blocks = nn.ModuleList(blocks)

        width = self.dec_blocks[0].in_channels
        self.bias = nn.Parameter(torch.zeros(1, width, 1, 1))

        self._init_weights()

    def _init_weights(self):
        total_blocks = len([1 for b in self.dec_blocks if hasattr(b, 'residual_scaling')])
        for block in self.dec_blocks:
            if hasattr(block, 'residual_scaling'):
                block.residual_scaling(total_blocks)

    def forward(self, enc_features, get_latents=False):
        stats = []
        min_res = min(enc_features.keys())
        feature = self.bias.expand(enc_features[min_res].shape)
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_train'):
                res = int(feature.shape[2])
                f_enc = enc_features[res]
                feature, block_stats = block.forward_train(feature, f_enc, get_latents=get_latents)
                stats.append(block_stats)
            else:
                feature = block(feature)
        return feature, stats

    def forward_uncond(self, nhw_repeat=(1, 1, 1), t=1.0):
        nB, nH, nW = nhw_repeat
        feature = self.bias.expand(nB, -1, nH, nW)
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_uncond'):
                feature = block.forward_uncond(feature, t)
            else:
                feature = block(feature)
        return feature

    def forward_with_latents(self, latents, nhw_repeat=None, t=1.0, paint_box=None):
        if nhw_repeat is None:
            nB, _, nH, nW = latents[0].shape
            feature = self.bias.expand(nB, -1, nH, nW)
        else:
            nB, nH, nW = nhw_repeat
            feature = self.bias.expand(nB, -1, nH, nW)
        idx = 0
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'forward_uncond'):
                feature = block.forward_uncond(feature, t, latent=latents[idx], paint_box=paint_box)
                idx += 1
            else:
                feature = block(feature)
        return feature

    def update(self):
        for block in self.dec_blocks:
            if hasattr(block, 'update'):
                block.update()

    def compress(self, enc_features):
        min_res = min(enc_features.keys())
        feature = self.bias.expand(enc_features[min_res].shape)
        strings_all = []

        de_features_list = []
        save_indices = [0, 3, 7, 11, 15]
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'compress'):
                res = feature.shape[2]
                f_enc = enc_features[res]
                feature, strs_batch = block.compress(feature, f_enc)
                strings_all.append(strs_batch)
            else:
                feature = block(feature)
            if i in save_indices:
                de_features_list.append(feature)
        return strings_all, feature, de_features_list

    def decompress(self, compressed_object: list):
        smallest_shape = compressed_object[-1]
        feature = self.bias.expand(smallest_shape)

        features_list = []
        save_indices = [3, 7, 11, 14, 15]
        str_i = 0
        for i, block in enumerate(self.dec_blocks):
            if hasattr(block, 'decompress'):
                strs_batch = compressed_object[str_i]
                str_i += 1
                feature = block.decompress(feature, strs_batch)
            else:
                feature = block(feature)

            if i in save_indices:
                features_list.append(feature)

        assert str_i == len(compressed_object) - 1, f'decoded={str_i}, len={len(compressed_object)}'
        return feature, features_list

class HierarchicalVAE(nn.Module):
    log2_e = math.log2(math.e)

    def __init__(self, config: dict):
        super().__init__()
        self.encoder = BottomUpEncoder(blocks=config.pop('enc_blocks'))
        self.decoder = TopDownDecoder(blocks=config.pop('dec_blocks'))
        self.out_net = config.pop('out_net')

        self.im_shift = float(config['im_shift'])
        self.im_scale = float(config['im_scale'])
        self.max_stride = config['max_stride']

        self.register_buffer('_dummy', torch.zeros(1), persistent=False)
        self._dummy: torch.Tensor

        self._stats_log = dict()
        self._flops_mode = False
        self.compressing = False

    def preprocess_input(self, im: torch.Tensor):
        assert (im.shape[2] % self.max_stride == 0) and (im.shape[3] % self.max_stride == 0)
        if not self._flops_mode:
            assert (im.dim() == 4) and (0 <= im.min() <= im.max() <= 1) and not im.requires_grad
        x = (im + self.im_shift) * self.im_scale
        return x

    def process_output(self, x: torch.Tensor):
        assert not x.requires_grad
        im_hat = x.clone().clamp_(min=-1.0, max=1.0).mul_(0.5).add_(0.5)
        return im_hat

    def preprocess_target(self, im: torch.Tensor):
        if not self._flops_mode:
            assert (im.dim() == 4) and (0 <= im.min() <= im.max() <= 1) and not im.requires_grad
        x = (im - 0.5) * 2.0
        return x

    def forward(self, im, return_rec=False):
        im = im.to(self._dummy.device)
        x = self.preprocess_input(im)
        x_target = self.preprocess_target(im)

        enc_features = self.encoder(x)
        feature, stats_all = self.decoder(enc_features)
        out_loss, x_hat = self.out_net.forward_loss(feature, x_target)

        if self._flops_mode:
            return x_hat

        nB, imC, imH, imW = im.shape
        kl_divergences = [stat['kl'].sum(dim=(1, 2, 3)) for stat in stats_all]
        ndims = imC * imH * imW
        kl = sum(kl_divergences) / ndims
        loss = (kl + out_loss).mean(0)

        with torch.no_grad():
            nats_per_dim = kl.detach().cpu().mean(0).item()
            im_hat = self.process_output(x_hat.detach())
            im_mse = tnf.mse_loss(im_hat, im, reduction='mean')
            psnr = -10 * math.log10(im_mse.item())
            kls = torch.stack([kl.mean(0) / ndims for kl in kl_divergences], dim=0)
            bpdim = kls * self.log2_e
            mode = 'train' if self.training else 'eval'
            self._stats_log[f'{mode}_bpdim'] = bpdim.tolist()
            self._stats_log[f'{mode}_bppix'] = (bpdim * imC).tolist()
            channel_bpps = [stat['kl'].sum(dim=(2,3)).mean(0).cpu() / (imH * imW) for stat in stats_all]
            self._stats_log[f'{mode}_channels'] = [(bpps*self.log2_e).tolist() for bpps in channel_bpps]

        stats = OrderedDict()
        stats['loss']  = loss
        stats['kl']    = nats_per_dim
        stats[self.out_net.loss_name] = out_loss.detach().cpu().mean(0).item()
        stats['bppix'] = nats_per_dim * self.log2_e * imC
        stats['psnr']  = psnr
        if return_rec:
            stats['im_hat'] = im_hat
        return stats

    @torch.no_grad()
    def forward_eval(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def uncond_sample(self, nhw_repeat, temprature=1.0):
        feature = self.decoder.forward_uncond(nhw_repeat, t=temprature)
        x_samples = self.out_net.sample(feature, temprature=temprature)
        im_samples = self.process_output(x_samples)
        return im_samples

    @torch.no_grad()
    def cond_sample(self, latents, nhw_repeat=None, temprature=1.0, paint_box=None):
        feature = self.decoder.forward_with_latents(latents, nhw_repeat, t=temprature, paint_box=paint_box)
        x_samples = self.out_net.sample(feature, temprature=temprature)
        im_samples = self.process_output(x_samples)
        return im_samples

    def forward_get_latents(self, im):
        x = self.preprocess_input(im)
        activations = self.encoder.forward(x)
        _, stats = self.decoder.forward(activations, get_latents=True)
        return stats

    @torch.no_grad()
    def inpaint(self, im, paint_box, steps=1, temprature=1.0):
        nB, imC, imH, imW = im.shape
        x1, y1, x2, y2 = paint_box
        h_slice = slice(round(y1*imH), round(y2*imH))
        w_slice = slice(round(x1*imW), round(x2*imW))
        im_input = im.clone()
        for i in range(steps):
            stats_all = self.forward_get_latents(im_input)
            latents = [st['z'] for st in stats_all]
            im_sample = self.cond_sample(latents, temprature=temprature, paint_box=paint_box)
            torch.clamp_(im_sample, min=0, max=1)
            im_input = im.clone()
            im_input[:, :, h_slice, w_slice] = im_sample[:, :, h_slice, w_slice]
        return im_sample

    def compress_mode(self, mode=True):
        if mode:
            self.decoder.update()
            if hasattr(self.out_net, 'compress'):
                self.out_net.update()
        self.compressing = mode

    @torch.no_grad()
    def compress(self, im):
        x = self.preprocess_input(im)
        enc_features = self.encoder(x)
        compressed_obj, feature, de_features_list = self.decoder.compress(enc_features)
        min_res = min(enc_features.keys())
        compressed_obj.append(tuple(enc_features[min_res].shape))
        if hasattr(self.out_net, 'compress'):
            x_tgt = self.preprocess_target(im)
            final_str = self.out_net.compress(feature, x_tgt)
            compressed_obj.append(final_str)
        return compressed_obj

    @torch.no_grad()
    def decompress(self, compressed_object):
        if hasattr(self.out_net, 'compress'):
            feature = self.decoder.decompress(compressed_object[:-1])
            x_hat = self.out_net.decompress(feature, compressed_object[-1])
        else:
            feature, features_list = self.decoder.decompress(compressed_object)
            x_hat = self.out_net.mean(feature)
        im_hat = self.process_output(x_hat)
        return im_hat, features_list

    @torch.no_grad()
    def compress_file(self, im, output_path):
        compressed_obj = self.compress(im)

        compressed_obj.append((2048, 2048))
        with open(output_path, 'wb') as f:
            pickle.dump(compressed_obj, file=f)

    @torch.no_grad()
    def decompress_file(self, bits_path):
        with open(bits_path, 'rb') as f:
            compressed_obj = pickle.load(file=f)
        img_h, img_w = compressed_obj.pop()
        im_hat, features_list = self.decompress(compressed_obj)
        return im_hat[:, :, :img_h, :img_w], features_list
def pad_divisible_by(img, div=64):
    h_old, w_old = img.height, img.width
    if (h_old % div == 0) and (w_old % div == 0):
        return img
    h_tgt = round(div * math.ceil(h_old / div))
    w_tgt = round(div * math.ceil(w_old / div))
    padded = tvf.pad(img, padding=padding, padding_mode='edge')
    return padded