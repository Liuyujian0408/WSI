import torch
from torch.hub import load_state_dict_from_url

from lvae.models.registry import register_model
import lvae.models.common as common
import lvae.models.qresvae.model as qres


@register_model
def qres34m(lmb=32, pretrained=False):
    cfg = dict()

    enc_nums = [6, 6, 6, 4, 2]
    dec_nums = [1, 2, 3, 3, 3]
    z_dims = [16, 14, 12, 10, 8]

    im_channels = 3
    ch = 96 
    cfg['enc_blocks'] = [
        common.patch_downsample(im_channels, ch*2, rate=4),
        *[qres.MyConvNeXtBlock(ch*2, kernel_size=7) for _ in range(enc_nums[0])], 
        qres.MyConvNeXtPatchDown(ch*2, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=7) for _ in range(enc_nums[1])], 
        qres.MyConvNeXtPatchDown(ch*4, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=5) for _ in range(enc_nums[2])], 
        qres.MyConvNeXtPatchDown(ch*4, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=3) for _ in range(enc_nums[3])], 
        qres.MyConvNeXtPatchDown(ch*4, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=1) for _ in range(enc_nums[4])], 
    ]
    cfg['dec_blocks'] = [
        *[qres.QLatentBlockX(ch*4, z_dims[0], kernel_size=1) for _ in range(dec_nums[0])], 
        common.patch_upsample(ch*4, ch*4, rate=2),
        *[qres.QLatentBlockX(ch*4, z_dims[1], kernel_size=3) for _ in range(dec_nums[1])],
        common.patch_upsample(ch*4, ch*4, rate=2),
        *[qres.QLatentBlockX(ch*4, z_dims[2], kernel_size=5) for _ in range(dec_nums[2])], 
        common.patch_upsample(ch*4, ch*4, rate=2),
        *[qres.QLatentBlockX(ch*4, z_dims[3], kernel_size=7) for _ in range(dec_nums[3])], 
        common.patch_upsample(ch*4, ch*2, rate=2),
        *[qres.QLatentBlockX(ch*2, z_dims[4], kernel_size=7) for _ in range(dec_nums[4])], 
        common.patch_upsample(ch*2, im_channels, rate=4)
    ]
    cfg['out_net'] = qres.MSEOutputNet(mse_lmb=lmb)

    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    cfg['max_stride'] = 64

    model = qres.HierarchicalVAE(cfg)
    if (pretrained is True) and (lmb in {16, 32, 64, 128, 256, 512, 1024, 2048}):
        url = f'https://huggingface.co/duanzh0/my-model-weights/resolve/main/qres34m/qres34m-lmb{lmb}.pt'
        msd = load_state_dict_from_url(url)['model']
        model.load_state_dict(msd)
    elif isinstance(pretrained, str):
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    else:
        assert pretrained is False, f'Invalid {pretrained=} and {lmb=}'
    return model

