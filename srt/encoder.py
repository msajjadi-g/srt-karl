# reviewed

import numpy as np
import torch
import torch.nn as nn
from srt.layers import RayEncoder, Transformer


class SRTConvBlock(nn.Module):
    def __init__(self, idim, hdim=None, odim=None):
        super().__init__()
        if hdim is None:
            hdim = idim

        if odim is None:
            odim = 2 * hdim

        conv_kwargs = {'bias': False, 'kernel_size': 3, 'padding': 1}
        self.layers = nn.Sequential(
            nn.Conv2d(idim, hdim, stride=1, **conv_kwargs),
            nn.ReLU(),
            nn.Conv2d(hdim, odim, stride=2, **conv_kwargs),
            nn.ReLU())
    
    def forward(self, x):
        return self.layers(x)


class SRTEncoder(nn.Module):
    def __init__(self, num_conv_blocks=4, num_att_blocks=10, pos_start_octave=0):
        super().__init__()
        self.ray_encoder = RayEncoder(pos_octaves=15, pos_start_octave=pos_start_octave,
                                      ray_octaves=15)

        conv_blocks = [SRTConvBlock(idim=183, hdim=96)]  # idim is 180 for us -- we don't pass the raw XYZ values in the posenc 
function
        cur_hdim = 96
        for i in range(1, num_conv_blocks):
            if cur_hdim < 1536:
                cur_hdim *= 2
            # no-op for our configs, but we always double the features for every layer
            conv_blocks.append(SRTConvBlock(idim=cur_hdim, odim=None))
        self.conv_blocks = nn.Sequential(*conv_blocks)

        self.per_patch_linear = nn.Conv2d(1536, 768, kernel_size=1)

        self.pixel_embedding = nn.Parameter(torch.randn(1, 768, 15, 20))
        # FWIW, we initialize with stddev=1/math.sqrt(d). Given that model initialization likely also differs between torch & jax, I 
don't think this needs to be addressed
        self.canonical_camera_embedding = nn.Parameter(torch.randn(1, 1, 768))
        self.non_canonical_camera_embedding = nn.Parameter(torch.randn(1, 1, 768))

        # this one's interesting. SRT as in the CVPR paper does not use actual self attention, but a special type:
        # the current features in the Nth layer don't self-attend, but they always attend into the initial patch embedding
        # (i.e., the output of the CNN). I think in your code, you get that behavior of you disable self-attention, but then call
        # transformer(x, x) regardless.
        # further, we used post-normalization rather than pre-normalization.
        # since then though, in OSRT, we found pre-norm and actual self-attention to perform better overall.
        # I think it's fine to use what works better here, but it may be less stable under some circumstances.
        self.transformer = Transformer(768, depth=num_att_blocks, heads=12, dim_head=64,
                                       mlp_dim=1536, selfatt=True)

    def forward(self, images, camera_pos, rays):
        """
        Args:
            images: [batch_size, num_images, 3, height, width]. Assume the first image is canonical.  # just in case: the input 
images should be shuffled in the data reader
            camera_pos: [batch_size, num_images, 3]
            rays: [batch_size, num_images, height, width, 3]
        Returns:
            scene representation: [batch_size, num_patches, channels_per_patch]
        """

        batch_size, num_images = images.shape[:2]

        x = images.flatten(0, 1)
        camera_pos = camera_pos.flatten(0, 1)
        rays = rays.flatten(0, 1)

        canonical_idxs = torch.zeros(batch_size, num_images)
        canonical_idxs[:, 0] = 1
        canonical_idxs = canonical_idxs.flatten(0, 1).unsqueeze(-1).unsqueeze(-1).to(x)
        camera_id_embedding = canonical_idxs * self.canonical_camera_embedding + \
                (1. - canonical_idxs) * self.non_canonical_camera_embedding

        ray_enc = self.ray_encoder(camera_pos, rays)
        x = torch.cat((x, ray_enc), 1)
        x = self.conv_blocks(x)
        x = self.per_patch_linear(x)
        height, width = x.shape[2:]
        x = x + self.pixel_embedding[:, :, :height, :width]
        x = x.flatten(2, 3).permute(0, 2, 1)
        x = x + camera_id_embedding

        patches_per_image, channels_per_patch = x.shape[1:]
        x = x.reshape(batch_size, num_images * patches_per_image, channels_per_patch)

        x = self.transformer(x)

        return x

