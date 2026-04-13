import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.densenet import densenet121

from timm.models import create_model

import numpy as np


def _load_weights_into_module(module, path):
    """Load .safetensors or .pth/.pt state dict into module; logs and skips if missing."""
    if not path or not isinstance(path, str) or not path.strip():
        return
    path = path.strip()
    if not os.path.isfile(path):
        logging.warning("ConvNeXt pretrained path not found (skip load): %s", path)
        return
    try:
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(path)
        else:
            ckpt = torch.load(path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and "model" in ckpt:
                state = ckpt["model"]
            else:
                state = ckpt
    except ImportError as e:
        logging.error(
            "Loading %s requires safetensors (pip install safetensors): %s", path, e
        )
        return
    target = module.state_dict()

    def _strip_prefixes(key):
        for p in ("model.", "module.", "backbone.", "encoder."):
            if key.startswith(p):
                return key[len(p) :]
        return key

    remapped = {}
    for k, v in state.items():
        nk = _strip_prefixes(k)
        if nk in target and target[nk].shape == v.shape:
            remapped[nk] = v
        elif k in target and target[k].shape == v.shape:
            remapped[k] = v

    if not remapped:
        remapped = state

    missing, unexpected = module.load_state_dict(remapped, strict=False)
    logging.info(
        "Loaded backbone weights from %s (matched tensors=%d, missing=%d, unexpected=%d)",
        path,
        len(remapped),
        len(missing),
        len(unexpected),
    )
    print(
        f"ConvNeXt backbone weights loaded successfully from: {path} "
        f"(matched tensors={len(remapped)}, missing={len(missing)}, unexpected={len(unexpected)})"
    )

class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalModel(nn.Module):
    def __init__(self, config, guidance=False):
        super(ConditionalModel, self).__init__()
        n_steps = config.diffusion.timesteps + 1
        data_dim = config.model.data_dim
        y_dim = config.data.num_classes
        arch = config.model.arch
        feature_dim = config.model.feature_dim
        hidden_dim = config.model.hidden_dim
        self.guidance = guidance
        # encoder for x
        if arch.startswith('convnextv2'):
            ckpt_path = getattr(config.model, 'convnextv2_pretrained_path', None)
            self.encoder_x = ConvNeXtV2Encoder(
                arch=arch, feature_dim=feature_dim, pretrained_path=ckpt_path
            )
        else:
            self.encoder_x = ResNetEncoder(arch=arch, feature_dim=feature_dim)
        # batch norm layer
        self.norm = nn.BatchNorm1d(feature_dim)

        # Unet
        if self.guidance:
            self.lin1 = ConditionalLinear(y_dim * 2, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)
        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(self, x, y, t, yhat=None):
        x = self.encoder_x(x)
        x = self.norm(x)
        if self.guidance:
            #for yh in yhat:
            y = torch.cat([y, yhat], dim=-1)
        y = self.lin1(y, t)
        y = self.unetnorm1(y)
        y = F.softplus(y)
        y = x * y
        y = self.lin2(y, t)
        y = self.unetnorm2(y)
        y = F.softplus(y)
        y = self.lin3(y, t)
        y = self.unetnorm3(y)
        y = F.softplus(y)
        y = self.lin4(y)

        return y



# ---------------------------------------------------------------------------
# ConvNeXt V2 encoder
# ---------------------------------------------------------------------------
class ConvNeXtV2Encoder(nn.Module):
    """
    timm convnextv2_* with num_classes=0 (feature vector, no classifier head).
    Optional pretrained_path: .safetensors or PyTorch .pth weights for the backbone only.
    Projection head self.g stays randomly initialized.
    """
    def __init__(self, arch='convnextv2_tiny', feature_dim=128, pretrained_path=None):
        super(ConvNeXtV2Encoder, self).__init__()
        backbone = create_model(arch, pretrained=False, num_classes=0)
        _load_weights_into_module(backbone, pretrained_path)
        self.featdim = backbone.num_features          # 768 for tiny
        self.backbone = backbone
        self.g = nn.Linear(self.featdim, feature_dim) # project to feature_dim

    def forward(self, x):
        feat = self.backbone(x)   # (B, featdim)
        feat = self.g(feat)       # (B, feature_dim)
        return feat


# ResNet 18/50, DenseNet-121, or ViT-style backbone as image encoder
class ResNetEncoder(nn.Module):
    def __init__(self, arch='resnet18', feature_dim=128):
        super(ResNetEncoder, self).__init__()

        self._densenet_pool = False
        if arch == 'resnet50':
            backbone = resnet50()
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'resnet18':
            backbone = resnet18()
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'densenet121':
            backbone = densenet121(pretrained=True)
            self.featdim = backbone.classifier.in_features
            # Only `features`; pool + flatten must match torchvision DenseNet.forward (no classifier).
            self.f = backbone.features
            self._densenet_pool = True
        elif arch == 'vit':
            backbone = create_model('pvt_v2_b2',
            pretrained=True,
            num_classes=4,
            drop_rate=0,
            drop_path_rate=0.1,
            drop_block_rate=None,
            )
            backbone.head = nn.Sequential()
            self.featdim = 512
        else:
            raise NotImplementedError(f'ResNetEncoder arch {arch!r} not supported.')

        if not self._densenet_pool:
            parts = []
            for name, module in backbone.named_children():
                if name != 'fc':
                    parts.append(module)
            self.f = nn.Sequential(*parts)

        self.g = nn.Linear(self.featdim, feature_dim)

    def forward_feature(self, x):
        feature = self.f(x)
        if self._densenet_pool:
            feature = F.relu(feature, inplace=True)
            feature = F.adaptive_avg_pool2d(feature, (1, 1))
        elif feature.ndim == 4:
            # Backbones like PVT/ViT variants can emit spatial maps; pool to vector.
            feature = F.adaptive_avg_pool2d(feature, (1, 1))
        elif feature.ndim == 3:
            # Token sequence output (B, N, C) -> global token average (B, C).
            feature = feature.mean(dim=1)
        feature = torch.flatten(feature, start_dim=1)
        feature = self.g(feature)

        return feature

    def forward(self, x):
        feature = self.forward_feature(x)
        return feature

