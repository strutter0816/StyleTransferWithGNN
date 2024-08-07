import torch
import torch.nn as nn
from functools import partial
import math
import numpy as np

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

from timm.models import create_model
from timm.models.vision_transformer import _cfg, Block
from .ms_deform_attn_func import MSDeformAttnFunction

def calc_padding(x_shape, patchsize, stride):
    xdim = x_shape
    padvert = -(xdim[0] - patchsize) % stride
    padhorz = -(xdim[1] - patchsize) % stride
    padtop = int(np.floor(padvert / 2.0))
    padbottom = int(np.ceil(padvert / 2.0))
    padleft = int(np.floor(padhorz / 2.0))
    padright = int(np.ceil(padhorz / 2.0))
    return padtop, padbottom, padleft, padright


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, patch_count=14, patch_stride=16, in_chans=3, embed_dim=768, with_norm=False):
        super().__init__()  

        # patch_stride = img_size // patch_count
        # patch_pad = (patch_stride * (patch_count - 1) + patch_size - img_size) // 2
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # num_patches = patch_count * patch_count
        # self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        # self.num_patches = num_patches
        # self.patch_count = patch_count
        # print(img_size, patch_count, patch_stride, patch_pad, patch_size)
        #    (64, 64)   23           3             2          (3, 3)
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=0)
        )
        
        if with_norm:
            self.norm = nn.LayerNorm(embed_dim)
    

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        padtop, padbottom, padleft, padright = calc_padding((H,W), self.patch_size[0], self.patch_stride)
        assert H == W, "only support H == W image"
        self.patch_count = math.floor((H + padtop + padbottom - (self.patch_size[0]-1) - 1) / self.patch_stride + 1)
        x = nn.functional.pad(x, (padleft, padright, padtop, padbottom))
        x = self.proj(x).flatten(2).transpose(1, 2)
        if hasattr(self, "norm"):
            x = self.norm(x)
        #assert x.shape[1] == self.num_patches
        return x


class Simple_Patch(nn.Module):
    def __init__(self, offset_embed, img_size=224, patch_size=16, patch_pixel=16, patch_count=14, patch_stride=16,
                 in_chans=3, embed_dim=192, another_linear=False, use_GE=False, local_feature=False, with_norm=False):
        super().__init__()
        self.H, self.W = patch_count, patch_count
        self.num_patches = patch_count * patch_count
        self.another_linear = another_linear
        self.content_style_embedding = nn.Sequential(
            nn.Conv2d(in_chans*2, in_chans, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1)
        )
        if self.another_linear:
            self.patch_embed = PatchEmbed(img_size, 1 if local_feature else patch_size, patch_count, patch_stride, in_chans, embed_dim, with_norm=with_norm)
            self.act = nn.GELU() if use_GE else nn.Identity()
            self.offset_predictor = nn.Linear(embed_dim, offset_embed, bias=False)
        else:
            self.patch_embed = PatchEmbed(img_size, 1 if local_feature else patch_size, patch_count, patch_stride, in_chans, offset_embed)

        self.img_size, self.patch_size, self.patch_stride, self.patch_pixel, self.patch_count = \
            img_size, patch_size, patch_stride, patch_pixel, patch_count
        self.in_chans, self.embed_dim = in_chans, embed_dim

    def reset_offset(self):
        if self.another_linear:
            nn.init.constant_(self.offset_predictor.weight, 0)
            if hasattr(self.offset_predictor, "bias") and self.offset_predictor.bias is not None:
                nn.init.constant_(self.offset_predictor.bias, 0)
        else:
            nn.init.constant_(self.patch_embed.proj.weight, 0)
            if hasattr(self.patch_embed.proj, "bias") and self.patch_embed.proj.bias is not None:
                nn.init.constant_(self.patch_embed.proj.bias, 0)
        print("Parameter for offsets reseted.")

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        #if x.dim() == 3:
        #    B, H, W = x.shape[0], self.img_size, self.img_size
        #    assert x.shape[1] == H * W
        #    x = x.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        if isinstance(x, tuple):
            img, content = x[0], x[1]
#             test
#             img = img.unsqueeze(0)
#             content = content.unsqueeze(0)
            x = self.content_style_embedding( torch.cat((img, content), dim=1) )
        else:
            img = x
        B, C, H, W = img.shape
        
        x = self.patch_embed(x)
        if self.another_linear:
            pred_offset = self.offset_predictor(self.act(x))
        else:
            pred_offset = x.contiguous()
        output_size = (self.patch_embed.patch_count, self.patch_embed.patch_count)
        return self.get_output(img, pred_offset, img_size=(H, W), output_size=output_size), output_size

class Simple_DePatch(Simple_Patch):
    def __init__(self, box_coder, show_dim=4, **kwargs):
        super().__init__(show_dim, **kwargs)
        self.box_coder = box_coder
        #self.register_buffer("value_spatial_shapes", torch.as_tensor([[self.img_size, self.img_size]], dtype=torch.long))
        self.register_buffer("value_level_start_index", torch.as_tensor([0], dtype=torch.long))
        self.output_proj = nn.Linear(self.in_chans * self.patch_pixel * self.patch_pixel, self.embed_dim)
        if kwargs["with_norm"]:
            self.with_norm=True
            self.norm = nn.LayerNorm(self.embed_dim)
        else:
            self.with_norm=False

    def get_output(self, img, pred_offset, img_size, output_size):
        #copyed
        B = img.shape[0]
        value_spatial_shapes = torch.as_tensor(img_size, dtype=torch.long, device=pred_offset.device).view(1, 2)
        num_sample_points = self.patch_pixel * self.patch_pixel * output_size[0] * output_size[1]

        sample_location = self.box_coder(pred_offset, img_size=img_size, output_size=output_size)
        sampling_locations = sample_location.view(B, num_sample_points,1,1,1,2).to(torch.float)
        attention_weights = torch.ones((B, num_sample_points, 1, 1, 1), device=img.device)
        
        x = img.view(B, self.in_chans, 1, -1).transpose(1, 3).contiguous()
        output = MSDeformAttnFunction.apply(x, value_spatial_shapes, self.value_level_start_index, sampling_locations, attention_weights, 1)
        # output_proj
        output = output.view(B, output_size[0]*output_size[1], self.in_chans*self.patch_pixel*self.patch_pixel)
        output = self.output_proj(output)
        if self.with_norm:
            output = self.norm(output)
        return output
