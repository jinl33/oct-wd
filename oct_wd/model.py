import torch
from generative.networks.nets import DiffusionModelUNet
from generative.networks.nets.diffusion_model_unet import get_timestep_embedding

from oct_wd import wavelet as wv

CONTEXT_DIM = 160


def build_unet(n_channels, context_dim=None):
    return DiffusionModelUNet(
        spatial_dims=3,
        in_channels=n_channels,
        out_channels=n_channels,
        num_channels=[40, 40, 80, 80, 160],
        norm_num_groups=8,
        attention_levels=[False, False, False, True, True],
        num_head_channels=[0, 0, 0, 16, 16],
        num_res_blocks=2,
        use_flash_attention=False,
        with_conditioning=context_dim is not None,
        cross_attention_dim=context_dim,
    )


class SloEncoder(torch.nn.Module):
    def __init__(self, dim=CONTEXT_DIM, widths=(32, 64, 128, 160), tokens=8):
        super().__init__()
        layers, c = [], 1
        for w in widths:
            layers += [
                torch.nn.Conv2d(c, w, 3, stride=2, padding=1),
                torch.nn.GroupNorm(8, w),
                torch.nn.SiLU(),
                torch.nn.Conv2d(w, w, 3, padding=1),
                torch.nn.GroupNorm(8, w),
                torch.nn.SiLU(),
            ]
            c = w
        self.body = torch.nn.Sequential(*layers)
        self.pool = torch.nn.AdaptiveAvgPool2d(tokens)
        self.proj = torch.nn.Linear(c, dim)
        self.pos = torch.nn.Parameter(torch.randn(tokens * tokens, dim) * 0.02)
        self.null = torch.nn.Parameter(torch.randn(tokens * tokens, dim) * 0.02)

    def empty(self, n):
        return self.null[None].expand(n, -1, -1)

    def forward(self, slo, keep=None):
        h = self.pool(self.body(slo[:, None]))
        h = self.proj(h.flatten(2).transpose(1, 2)) + self.pos
        if keep is not None:
            h = torch.where(keep[:, None, None], h, self.empty(h.shape[0]).to(h.dtype))
        return h


class SloDiffusion(torch.nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.unet = build_unet(n_channels, CONTEXT_DIM)
        self.slo = SloEncoder()

    def context(self, slo, n, keep=None):
        return self.slo.empty(n) if slo is None else self.slo(slo, keep)

    def forward(self, x, timesteps, slo=None, keep=None):
        return self.unet(x=x, timesteps=timesteps, context=self.context(slo, x.shape[0], keep))


def build_model(n_channels, slo_size=0):
    return SloDiffusion(n_channels) if slo_size else build_unet(n_channels)


def predict(model, x, t, slo=None):
    if isinstance(model, SloDiffusion):
        return model(x, t, slo)
    return model(x=x, timesteps=t)


class Encoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        unet = model.unet if isinstance(model, SloDiffusion) else model
        self.slo = model.slo if isinstance(model, SloDiffusion) else None
        self.base = unet.block_out_channels[0]
        self.time_embed = unet.time_embed
        self.conv_in = unet.conv_in
        self.down_blocks = unet.down_blocks
        self.middle_block = unet.middle_block
        self.dim = sum(unet.block_out_channels) + unet.block_out_channels[-1]

    def forward(self, x, t, slo=None):
        context = None
        if self.slo is not None:
            context = self.slo.empty(x.shape[0]) if slo is None else self.slo(slo)
        emb = self.time_embed(get_timestep_embedding(t, self.base).to(dtype=x.dtype))
        h = self.conv_in(x)
        pooled = []
        for block in self.down_blocks:
            h, _ = block(hidden_states=h, temb=emb, context=context)
            pooled.append(h.mean(dim=(2, 3, 4)))
        h = self.middle_block(hidden_states=h, temb=emb, context=context)
        pooled.append(h.mean(dim=(2, 3, 4)))
        return torch.cat(pooled, dim=1).float()


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    axes = wv.DESIGNS[ckpt["design"]]
    model = build_model(wv.channels(axes), ckpt.get("slo_size", 0))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt, axes


def band_scale(ckpt):
    band = ckpt.get("band_scale")
    return None if band is None else torch.tensor(band, dtype=torch.float32)
