import pickle
from collections import OrderedDict
from PIL import Image
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torch.distributions as td
import torchvision.transforms.functional as tvf
from compressai.entropy_models import GaussianConditional

import lvae.models.common as common
from lvae.models.entropy_coding import gaussian_log_prob_mass


class GaussianNLLOutputNet(nn.Module):
    def __init__(self, conv_mean, conv_scale, bin_size=1/127.5):
        super().__init__()
        self.conv_mean  = conv_mean
        self.conv_scale = conv_scale
        self.bin_size = bin_size
        self.loss_name = 'nll'

    def forward_loss(self, feature, x_tgt):
        feature = feature.float()
        p_mean = self.conv_mean(feature)
        p_logscale = self.conv_scale(feature)
        p_logscale = tnf.softplus(p_logscale + 16) - 16
        log_prob = gaussian_log_prob_mass(p_mean, torch.exp(p_logscale), x_tgt, bin_size=self.bin_size)
        assert log_prob.shape == x_tgt.shape
        nll = -log_prob.mean(dim=(1,2,3))
        return nll, p_mean

    def mean(self, feature):
        p_mean = self.conv_mean(feature)
        return p_mean

    def sample(self, feature, mode='continuous', temprature=None):
        p_mean = self.conv_mean(feature)
        p_logscale = self.conv_scale(feature)
        p_scale = torch.exp(p_logscale)
        if temprature is not None:
            p_scale = p_scale * temprature

        if mode == 'continuous':
            samples = p_mean + p_scale * torch.randn_like(p_mean)
        elif mode == 'discrete':
            raise NotImplementedError()
        else:
            raise ValueError()
        return samples

    def update(self):
        self.discrete_gaussian = GaussianConditional(None, scale_bound=0.11)
        device = next(self.parameters()).device
        self.discrete_gaussian = self.discrete_gaussian.to(device=device)
        lower = self.discrete_gaussian.lower_bound_scale.bound.item()
        max_scale = 20
        scale_table = torch.exp(torch.linspace(math.log(lower), math.log(max_scale), steps=128))
        updated = self.discrete_gaussian.update_scale_table(scale_table)
        self.discrete_gaussian.update()

    def _preapre_codec(self, feature, x=None):
        assert not feature.requires_grad
        pm = self.conv_mean(feature)
        pm = torch.round(pm * 127.5 + 127.5) / 127.5 - 1 
        plogv = self.conv_scale(feature)
        pm = pm / self.bin_size
        plogv = plogv - math.log(self.bin_size)
        if x is not None:
            x = x / self.bin_size
        return pm, plogv, x

    def compress(self, feature, x):
        pm, plogv, x = self._preapre_codec(feature, x)
        indexes = self.discrete_gaussian.build_indexes(torch.exp(plogv))
        strings = self.discrete_gaussian.compress(x, indexes, means=pm)
        return strings

    def decompress(self, feature, strings):
        pm, plogv, _ = self._preapre_codec(feature)
        indexes = self.discrete_gaussian.build_indexes(torch.exp(plogv))
        x_hat = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        x_hat = x_hat * self.bin_size
        return x_hat


class MSEOutputNet(nn.Module):
    def __init__(self, mse_lmb):
        super().__init__()
        self.mse_lmb = float(mse_lmb)
        self.loss_name = 'mse'

    def forward_loss(self, x_hat, x_tgt):
        assert x_hat.shape == x_tgt.shape
        mse = tnf.mse_loss(x_hat, x_tgt, reduction='none').mean(dim=(1,2,3))
        loss = mse * self.mse_lmb
        return loss, x_hat

    def mean(self, x_hat, temprature=None):
        return x_hat
    sample = mean


class VDBlock(nn.Module):
    def __init__(self, in_ch, hidden_ch=None, out_ch=None, residual=True,
                 use_3x3=True, zero_last=False):
        super().__init__()
        out_ch = out_ch or in_ch
        hidden_ch = hidden_ch or round(in_ch * 0.25)
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.residual = residual
        self.c1 = common.conv_k1s1(in_ch, hidden_ch)
        self.c2 = common.conv_k3s1(hidden_ch, hidden_ch) if use_3x3 else common.conv_k1s1(hidden_ch, hidden_ch)
        self.c3 = common.conv_k3s1(hidden_ch, hidden_ch) if use_3x3 else common.conv_k1s1(hidden_ch, hidden_ch)
        self.c4 = common.conv_k1s1(hidden_ch, out_ch, zero_weights=zero_last)

    def residual_scaling(self, N):
        self.c4.weight.data.mul_(math.sqrt(1 / N))

    def forward(self, x):
        xhat = self.c1(tnf.gelu(x))
        xhat = self.c2(tnf.gelu(xhat))
        xhat = self.c3(tnf.gelu(xhat))
        xhat = self.c4(tnf.gelu(xhat))
        out = (x + xhat) if self.residual else xhat
        return out

class VDBlockPatchDown(VDBlock):
    def __init__(self, in_ch, out_ch, down_rate=2):
        super().__init__(in_ch, residual=True)
        self.downsapmle = common.patch_downsample(in_ch, out_ch, rate=down_rate)

    def forward(self, x):
        x = super().forward(x)
        out = self.downsapmle(x)
        return out


from timm.models.convnext import ConvNeXtBlock
class MyConvNeXtBlock(ConvNeXtBlock):
    def __init__(self, dim, mlp_ratio=2, **kwargs):
        super().__init__(dim, mlp_ratio=mlp_ratio, **kwargs)
        self.norm.affine = True 

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1).contiguous()
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        x = self.drop_path(x) + shortcut
        return x

class MyConvNeXtPatchDown(MyConvNeXtBlock):
    def __init__(self, in_ch, out_ch, down_rate=2, mlp_ratio=2, kernel_size=7):
        super().__init__(in_ch, mlp_ratio=mlp_ratio, kernel_size=kernel_size)
        self.downsapmle = common.patch_downsample(in_ch, out_ch, rate=down_rate)

    def forward(self, x):
        x = super().forward(x)
        out = self.downsapmle(x)
        return out


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


class QLatentBlockX(nn.Module):
    def __init__(self, width, zdim, enc_width=None, kernel_size=7):
        super().__init__()
        self.in_channels  = width
        self.out_channels = width

        enc_width = enc_width or width
        hidden = int(max(width, enc_width) * 0.25)
        concat_ch = (width * 2) if enc_width is None else (width + enc_width)
        use_3x3 = (kernel_size >= 3)
        self.resnet_front = MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.resnet_end   = MyConvNeXtBlock(width, kernel_size=kernel_size)
        self.posterior = VDBlock(concat_ch, hidden, zdim, residual=False, use_3x3=use_3x3)
        self.prior     = VDBlock(width, hidden, zdim * 2, residual=False, use_3x3=use_3x3,
                                 zero_last=True)
        self.z_proj = nn.Sequential(
            common.conv_k3s1(zdim, hidden//2) if use_3x3 else common.conv_k1s1(zdim, hidden//2),
            nn.GELU(),
            common.conv_k1s1(hidden//2, width),
        )
        self.discrete_gaussian = GaussianConditional(None)

    def residual_scaling(self, N):
        self.z_proj[2].weight.data.mul_(math.sqrt(1 / 3*N))

    def transform_prior(self, feature):
        feature = self.resnet_front(feature)
        pm, plogv = self.prior(feature).chunk(2, dim=1)
        plogv = tnf.softplus(plogv + 2.3) - 2.3 
        return feature, pm, plogv

    def forward_train(self, feature, enc_feature, get_latents=False):
        feature, pm, plogv = self.transform_prior(feature)
        pv = torch.exp(plogv)
        assert feature.shape[2:4] == enc_feature.shape[2:4]
        qm = self.posterior(torch.cat([feature, enc_feature], dim=1))
        if self.training:
            z_sample = qm + torch.empty_like(qm).uniform_(-0.5, 0.5)
            log_prob = gaussian_log_prob_mass(pm, pv, x=z_sample, bin_size=1.0, prob_clamp=1e-6)
            kl = -1.0 * log_prob
        else:
            z_sample, probs = self.discrete_gaussian(qm, scales=pv, means=pm)
            kl = -1.0 * torch.log(probs)
        feature = feature + self.z_proj(z_sample)
        feature = self.resnet_end(feature)
        if get_latents:
            return feature, dict(z=z_sample.detach(), kl=kl)
        return feature, dict(kl=kl)

    def forward_uncond(self, feature, t=1.0, latent=None, paint_box=None):
        feature, pm, plogv = self.transform_prior(feature)
        pv = torch.exp(plogv)
        pv = pv * t
        if latent is None: 
            z = pm + pv * torch.randn_like(pm) + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
        elif paint_box is not None:
            nB, zC, zH, zW = latent.shape
            if min(zH, zW) == 1:
                z = latent
            else:
                x1, y1, x2, y2 = paint_box
                h_slice = slice(round(y1*zH), round(y2*zH))
                w_slice = slice(round(x1*zW), round(x2*zW))
                z_sample = pm + pv * torch.randn_like(pm) + torch.empty_like(pm).uniform_(-0.5, 0.5) * t
                z_patch = z_sample[:, :, h_slice, w_slice]
                z = torch.clone(latent)
                z[:, :, h_slice, w_slice] = z_patch
        else: 
            assert pm.shape == latent.shape
            z = latent
        feature = feature + self.z_proj(z)
        feature = self.resnet_end(feature)
        return feature

    def update(self):
        min_scale = 0.1
        max_scale = 20
        log_scales = torch.linspace(math.log(min_scale), math.log(max_scale), steps=64)
        scale_table = torch.exp(log_scales)
        updated = self.discrete_gaussian.update_scale_table(scale_table)
        self.discrete_gaussian.update()

    def compress(self, feature, enc_feature):
        feature, pm, plogv = self.transform_prior(feature)
        qm = self.posterior(torch.cat([feature, enc_feature], dim=1))
        indexes = self.discrete_gaussian.build_indexes(torch.exp(plogv))
        strings = self.discrete_gaussian.compress(qm, indexes, means=pm)
        zhat = self.discrete_gaussian.quantize(qm, mode='dequantize', means=pm)
        feature = feature + self.z_proj(zhat)
        feature = self.resnet_end(feature)
        return feature, strings

    def decompress(self, feature, strings):
        feature, pm, plogv = self.transform_prior(feature)
        indexes = self.discrete_gaussian.build_indexes(torch.exp(plogv))
        zhat = self.discrete_gaussian.decompress(strings, indexes, means=pm)
        feature = feature + self.z_proj(zhat)
        feature = self.resnet_end(feature)
        return feature

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
        img = Image.open(im)
        img_padded = pad_divisible_by(img, div=self.max_stride)
        device = next(self.parameters()).device
        im = tvf.to_tensor(img_padded).unsqueeze_(0).to(device=device)
        compressed_obj = self.compress(im)

        compressed_obj.append((4096, 4096))
        with open(output_path, 'wb') as f:
            pickle.dump(compressed_obj, file=f)

    @torch.no_grad()
    def decompress_file(self, bits_path):
        with open(bits_path, 'rb') as f:
            compressed_obj = pickle.load(file=f)
        img_h, img_w = compressed_obj.pop()
        im_hat, features_list = self.decompress(compressed_obj)
        return im_hat[:, :, :img_h, :img_w], features_list

    @torch.no_grad()
    def decompress_batch_from_memory(self, compressed_objs: list, image_sizes: list):
        im_hat, new_feature = self.decompress(compressed_objs)
        imgs = []
        for i, (h, w) in enumerate(image_sizes):
            imgs.append(im_hat[i:i+1, :, :h, :w])
        return torch.cat(imgs, dim=0), new_feature



def pad_divisible_by(img, div=64):
    h_old, w_old = img.height, img.width
    if (h_old % div == 0) and (w_old % div == 0):
        return img
    h_tgt = round(div * math.ceil(h_old / div))
    w_tgt = round(div * math.ceil(w_old / div))
    padding = (0, 0, (w_tgt - w_old), (h_tgt - h_old))
    padded = tvf.pad(img, padding=padding, padding_mode='edge')
    return padded
