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


class FiLMThreeWayFusion(nn.Module):
    def __init__(self, y_dim):
        super(FiLMThreeWayFusion, self).__init__()
        self.to_gamma = nn.Linear(y_dim, y_dim)
        self.to_beta = nn.Linear(y_dim, y_dim)
        self.gate = nn.Linear(y_dim * 2, y_dim)
        self.out_proj = nn.Linear(y_dim * 3, y_dim * 3)

    def forward(self, y_t_g, y_t_shared, y_t_l):
        # Global-conditioned FiLM parameters modulate local stream.
        gamma = self.to_gamma(y_t_g)
        beta = self.to_beta(y_t_g)
        y_t_l_mod = gamma * y_t_l + beta

        # Gate mixes the modulated local stream with shared noisy latent.
        gate = torch.sigmoid(self.gate(torch.cat([y_t_l_mod, y_t_shared], dim=-1)))
        y_shared_local = gate * y_t_l_mod + (1.0 - gate) * y_t_shared

        # Keep output size equal to original three-way concatenation size.
        fused = torch.cat([y_t_g, y_shared_local, y_t_l_mod], dim=-1)
        return self.out_proj(fused)


class GatedInferenceFusion(nn.Module):
    def __init__(self, y_dim):
        super(GatedInferenceFusion, self).__init__()
        self.weight_head = nn.Linear(y_dim * 2, 2)
        self.yt_proj = nn.Linear(y_dim, y_dim)
        self.merge = nn.Linear(y_dim * 2, y_dim)
        self.out_proj = nn.Linear(y_dim * 3, y_dim * 3)

    def forward(self, y_g, y_t, y_l):
        # Predict per-sample softmax weights for global vs local guidance.
        logits = self.weight_head(torch.cat([y_g, y_l], dim=-1))
        weights = torch.softmax(logits, dim=-1)
        y_guided = weights[:, :1] * y_g + weights[:, 1:] * y_l

        # Lightweight projection combines weighted guidance with current noisy latent.
        y_t_feat = self.yt_proj(y_t)
        y_mix = self.merge(torch.cat([y_guided, y_t_feat], dim=-1))

        # Keep output size equal to original inference concatenation size.
        fused = torch.cat([y_guided, y_t_feat, y_mix], dim=-1)
        return self.out_proj(fused)


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
            self.film_three_way_fusion = FiLMThreeWayFusion(y_dim)
            self.gated_inference_fusion = GatedInferenceFusion(y_dim)
            self.lin1 = ConditionalLinear(y_dim * 3, feature_dim, n_steps)
        else:
            self.lin1 = ConditionalLinear(y_dim, feature_dim, n_steps)
        self.unetnorm1 = nn.BatchNorm1d(feature_dim)
        self.lin2 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm2 = nn.BatchNorm1d(feature_dim)
        self.lin3 = ConditionalLinear(feature_dim, feature_dim, n_steps)
        self.unetnorm3 = nn.BatchNorm1d(feature_dim)
        self.lin4 = nn.Linear(feature_dim, y_dim)

    def forward(
        self,
        x,
        y,
        t,
        yhat=None,
        yhat_global=None,
        yhat_local=None,
        use_inference_fusion=False,
    ):
        x = self.encoder_x(x)
        x = self.norm(x)
        if self.guidance:
            # Backward-compatible fallback when callers only provide a single guidance tensor.
            if yhat_global is None:
                yhat_global = yhat
            if yhat_local is None:
                yhat_local = yhat
            if yhat_global is None or yhat_local is None:
                raise ValueError(
                    "Guidance enabled but no guidance tensor provided. "
                    "Pass yhat or (yhat_global, yhat_local)."
                )

            if use_inference_fusion:
                y = self.gated_inference_fusion(yhat_global, y, yhat_local)
            else:
                y = self.film_three_way_fusion(yhat_global, y, yhat_local)
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


# ResNet 18 or 50 as image encoder
class ResNetEncoder(nn.Module):
    def __init__(self, arch='resnet18', feature_dim=128):
        super(ResNetEncoder, self).__init__()

        self.f = []
        #print(arch)
        if arch == 'resnet50':
            backbone = resnet50()
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'resnet18':
            backbone = resnet18()
            self.featdim = backbone.fc.weight.shape[1]
        elif arch == 'densenet121':
            backbone = densenet121(pretrained=True)
            self.featdim = backbone.classifier.weight.shape[1]
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

        for name, module in backbone.named_children():
            #if not isinstance(module, nn.Linear):
            #    self.f.append(module)
            if name != 'fc':
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        
        #print(self.featdim)
        self.g = nn.Linear(self.featdim, feature_dim)
        #self.z = nn.Linear(feature_dim, 4)

    def forward_feature(self, x):
        feature = self.f(x)
        #x = x.mean(dim=1)

        feature = torch.flatten(feature, start_dim=1)
        feature = self.g(feature)

        return feature

    def forward(self, x):
        feature = self.forward_feature(x)
        return feature

