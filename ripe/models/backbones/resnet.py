# ResNet-18 backbone for RIPE.
#
# Same constructor arguments and outputs as ripe/models/backbones/vgg.py, so it is selected
# with `backbones=resnet` (conf/backbones/resnet.yaml) and the rest of RIPE (keypoint sampling,
# hypercolumn descriptors, matching, RANSAC, REINFORCE) is unchanged.
#
# Why this is not a direct swap of torchvision's ResNet-18:
#   RIPE's VGG-19 encoder returns 4 feature maps at strides 1, 2, 4, 8 (64/128/256/512 ch).
#   The keypoint heatmap is produced at the resolution of the shallowest map, so it must be
#   stride 1. ResNet-18 downsamples 4x in its first two ops (7x7 stride-2 conv + max-pool),
#   so it has no full-resolution feature map at all. We therefore add a small full-resolution
#   stem and take the deeper levels from the pretrained ResNet stages:
#
#     level 1: stem (2x conv3x3)   stride 1    64 ch   new, randomly initialised
#     level 2: ResNet layer1       stride 4    64 ch   ImageNet-pretrained
#     level 3: ResNet layer2       stride 8   128 ch   ImageNet-pretrained
#     level 4: ResNet layer3       stride 16  256 ch   ImageNet-pretrained
#
#   layer4 (512 ch, stride 32) is dropped by default, analogous to RIPE cutting VGG-19 before
#   its conv5 block. num_layers=5 keeps it (ablation). stem_channels=(16, 32) gives the
#   narrow-stem ablation (level 1 then has 32 ch, hypercolumn 480-d).
#   Hypercolumn: 64+64+128+256 = 512-d (VGG: 960-d), projected to 256-d by RIPE's 1x1 conv.
#
#   The VGG decoder has its channel widths hard-coded (512/256/128/64). Here the refiners are
#   built from the encoder dims, so the same code works for any number of levels / widths.

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from .backbone_base import BackboneBase
from .vgg_utils import ConvRefiner, Decoder


class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True, num_input_channels=3, num_layers=4, stem_channels=(32, 64)):
        super().__init__()
        assert num_layers in (4, 5), "num_layers must be 4 (stem+layer1-3) or 5 (stem+layer1-4)"
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = tvm.resnet18(weights=weights)

        # full-resolution stem: supplies the stride-1 features the heatmap needs
        # (default 32 -> 64 channels; the narrow-stem ablation uses 16 -> 32)
        c1, c2 = stem_channels
        self.stem = nn.Sequential(
            nn.Conv2d(num_input_channels, c1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )

        if num_input_channels != 3:
            # pretrained conv1 expects RGB; a different input needs a fresh first conv
            resnet.conv1 = nn.Conv2d(num_input_channels, 64, 7, 2, 3, bias=False)

        # Original ResNet input block, stride 4. It runs on the image in parallel to the stem
        # (not on top of it), so the pretrained layers see the input they were trained on.
        self.resnet_in = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        stages = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]
        self.stages = nn.ModuleList(stages[: num_layers - 1])

        self.dims = [c2, 64, 128, 256, 512][:num_layers]

    def get_dim_layers(self):
        return self.dims

    def forward(self, x, **kwargs):
        feats, sizes = [], []

        f = self.stem(x)  # stride 1
        feats.append(f)
        sizes.append(f.shape[-2:])

        f = self.resnet_in(x)  # stride 4
        for stage in self.stages:
            f = stage(f)
            feats.append(f)
            sizes.append(f.shape[-2:])

        return feats, sizes


def build_conv_refiners(dims, num_output_channels, hidden_blocks):
    """Coarse-to-fine decoder refiners with widths derived from the encoder dims.

    Same structure as VGG._create_conv_refiner, which hard-codes the VGG widths.
    Going deepest -> shallowest, each refiner gets [level features, context from the previous
    refiner] and outputs [num_output_channels logits, context sized for the next level].
    """
    rev = list(reversed(dims))  # deep -> shallow
    refiners = {}
    for i, d in enumerate(rev):
        # the deepest level has no incoming context; every later level gets a context
        # tensor with d channels from the refiner before it
        in_dim = d if i == 0 else 2 * d
        ctx_out = rev[i + 1] if i + 1 < len(rev) else 1
        # Keys only need to be unique and ordered deep -> shallow: Decoder iterates over them
        # in insertion order (VGG uses "8", "4", "2", "1").
        refiners[f"level{len(rev) - i}"] = ConvRefiner(
            in_dim,
            d,
            ctx_out + num_output_channels,
            hidden_blocks=hidden_blocks,
            residual=True,
        )
    return nn.ModuleDict(refiners)


class ResNet(BackboneBase):
    def __init__(
        self,
        nchannels=3,
        pretrained=True,
        use_instance_norm=True,
        mode="dect",
        num_layers=4,
        stem_channels=(32, 64),
    ):
        super().__init__(nchannels=nchannels, use_instance_norm=use_instance_norm)

        self.nchannels = nchannels
        self.mode = mode

        if self.mode not in ["dect", "desc", "dect+desc"]:
            raise ValueError("mode should be 'dect', 'desc' or 'dect+desc'")

        # encoder is built first because the decoder widths depend on it
        self.encoder = ResNet18Encoder(
            pretrained=pretrained,
            num_input_channels=nchannels,
            num_layers=num_layers,
            stem_channels=tuple(stem_channels),
        )

        NUM_OUTPUT_CHANNELS, hidden_blocks = self._get_mode_params(mode)
        conv_refiner = build_conv_refiners(self.encoder.get_dim_layers(), NUM_OUTPUT_CHANNELS, hidden_blocks)
        self.decoder = Decoder(conv_refiner, num_prototypes=NUM_OUTPUT_CHANNELS)

    def _get_mode_params(self, mode):
        # same output channels / refiner depth as the VGG backbone
        if mode == "dect":
            return 1, 8
        elif mode == "desc":
            return 256, 5
        elif mode == "dect+desc":
            return 256 + 1, 8

    def get_dim_layers_encoder(self):
        return self.encoder.get_dim_layers()

    def _forward(self, x):
        # Same coarse-to-fine loop as VGG._forward. Upsampling uses the stored feature sizes,
        # so the non-uniform stride jumps here (16 -> 8 -> 4 -> 1) work without changes.
        features, sizes = self.encoder(x)
        output = 0
        context = None
        scales = list(self.decoder.scales)
        for idx, (feature_map, scale) in enumerate(zip(reversed(features), scales)):
            delta_descriptor, context = self.decoder(feature_map, scale=scale, context=context)
            output = output + delta_descriptor
            if idx < len(scales) - 1:
                size = sizes[-(idx + 2)]
                output = F.interpolate(output, size=size, mode="bilinear", align_corners=False)
                context = F.interpolate(context, size=size, mode="bilinear", align_corners=False)

        if self.mode == "dect":
            return {"heatmap": output, "coarse_descs": features}
        elif self.mode == "desc":
            return {"fine_descs": output, "coarse_descs": features}
        elif self.mode == "dect+desc":
            logits = output[:, :1].contiguous()
            descs = output[:, 1:].contiguous()
            return {"heatmap": logits, "fine_descs": descs, "coarse_descs": features}
