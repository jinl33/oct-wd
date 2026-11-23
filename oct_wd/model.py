import torch
import torch.nn.functional as F
from generative.networks.nets import DiffusionModelUNet
from generative.networks.nets.diffusion_model_unet import get_timestep_embedding

from oct_wd import wavelet as wv

CONTEXT_DIM = 160
Z_DOWN = (1, 1, 1, 1)


class ZUpsample(torch.nn.Module):
    def __init__(self, up):
        super().__init__()
        self.conv = up.conv

    def forward(self, x, emb=None):
        dtype = x.dtype
        x = F.interpolate(x.float() if dtype == torch.bfloat16 else x, scale_factor=(1.0, 2.0, 2.0), mode="nearest").to(dtype)
        return self.conv(x)


def build_unet(n_channels, context_dim=None, z_down=Z_DOWN):
    unet = DiffusionModelUNet(
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
    if len(z_down) != len(unet.down_blocks) - 1:
        raise ValueError("z_down needs one 0/1 value per downsampling (%d)" % (len(unet.down_blocks) - 1))
    for i, down in enumerate(z_down):
        if not down:
            unet.down_blocks[i].downsampler.op.conv.stride = (1, 2, 2)
            up = unet.up_blocks[len(unet.up_blocks) - 2 - i]
            up.upsampler = ZUpsample(up.upsampler)
    return unet


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
        self.widths = tuple(widths)
        self.body = torch.nn.Sequential(*layers)
        self.pool = torch.nn.AdaptiveAvgPool2d(tokens)
        self.proj = torch.nn.Linear(c, dim)
        self.pos = torch.nn.Parameter(torch.randn(tokens * tokens, dim) * 0.02)
        self.null = torch.nn.Parameter(torch.randn(tokens * tokens, dim) * 0.02)

    def empty(self, n):
        return self.null[None].expand(n, -1, -1)

    def stages(self, slo):
        h, out = slo[:, None], []
        for k in range(len(self.widths)):
            h = self.body[6 * k: 6 * k + 6](h)
            out.append(h)
        return out

    def to_tokens(self, h, keep=None):
        h = self.pool(h)
        h = self.proj(h.flatten(2).transpose(1, 2)) + self.pos
        if keep is not None:
            h = torch.where(keep[:, None, None], h, self.empty(h.shape[0]).to(h.dtype))
        return h

    def forward(self, slo, keep=None):
        return self.to_tokens(self.body(slo[:, None]), keep)


class SloCrossAttention(torch.nn.Module):
    def __init__(self, channels, context_dim, inner=64, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = torch.nn.GroupNorm(8, channels)
        self.q = torch.nn.Linear(channels, inner, bias=False)
        self.kv = torch.nn.Linear(context_dim, 2 * inner, bias=False)
        self.out = torch.nn.Linear(inner, channels)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, h, context):
        b, c = h.shape[:2]
        x = self.norm(h).flatten(2).transpose(1, 2)
        q = self.q(x).view(b, x.shape[1], self.heads, -1).transpose(1, 2)
        k, v = self.kv(context.to(x.dtype)).view(b, context.shape[1], 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, x.shape[1], -1)
        return h + self.out(a).transpose(1, 2).reshape(h.shape)


class CrossInjection:
    def __init__(self, layers, context, inner=None):
        self.layers, self.context, self.inner = layers, context, inner

    def time(self, emb):
        return emb if self.inner is None else self.inner.time(emb)

    def __call__(self, level, h):
        if self.inner is not None:
            h = self.inner(level, h)
        key = str(level)
        return self.layers[key](h, self.context) if key in self.layers else h


class Injection:
    def __init__(self, adapter, maps, pooled, gate):
        self.adapter, self.maps, self.pooled, self.gate = adapter, maps, pooled, gate

    def time(self, emb):
        return emb + (self.adapter.temb(self.pooled) * self.gate[:, None]).to(emb.dtype)

    def __call__(self, level, h):
        m = F.adaptive_avg_pool2d(self.maps[self.adapter.stage_of[level]], (h.shape[2], h.shape[4]))
        add = self.adapter.proj[level](m) * self.gate[:, None, None, None]
        return h + add[:, :, :, None, :].to(h.dtype)


class SloAdapter(torch.nn.Module):
    def __init__(self, stage_widths, level_widths, temb_dim):
        super().__init__()
        self.stage_of = tuple(min(i + 1, len(stage_widths) - 1) for i in range(len(level_widths)))
        self.proj = torch.nn.ModuleList(torch.nn.Conv2d(stage_widths[s], c, 1, bias=False) for s, c in zip(self.stage_of, level_widths))
        self.temb = torch.nn.Linear(stage_widths[-1], temb_dim)
        for p in self.parameters():
            torch.nn.init.zeros_(p)

    def forward(self, stages, grid, keep=None):
        xy = grid[:, :2].permute(0, 2, 3, 1).contiguous()
        valid = grid[:, 2:3]
        maps = {}
        for s in sorted(set(self.stage_of)):
            f = stages[s]
            maps[s] = F.grid_sample(f, xy.to(f.dtype), mode="bilinear", padding_mode="zeros", align_corners=False) * valid.to(f.dtype)
        gate = torch.ones(grid.shape[0], device=grid.device) if keep is None else keep.float()
        return Injection(self, maps, stages[-1].mean(dim=(2, 3)), gate)


def condition(slo_encoder, adapter, slo, n, keep=None, xattn=None):
    if slo is None:
        context, inject = slo_encoder.empty(n), None
    elif adapter is None:
        image = slo[0] if isinstance(slo, (tuple, list)) else slo
        context, inject = slo_encoder(image, keep), None
    else:
        if not isinstance(slo, (tuple, list)) or len(slo) != 2:
            raise ValueError("this model needs the SLO together with its scan-line grid")
        image, grid = slo
        stages = slo_encoder.stages(image)
        context, inject = slo_encoder.to_tokens(stages[-1], keep), adapter(stages, grid, keep)
    if xattn is not None and len(xattn):
        inject = CrossInjection(xattn, context, inject)
    return context, inject


def slo_vector(slo_encoder, slo, n, device):
    if slo is None:
        return torch.zeros(n, slo_encoder.widths[-1], device=device)
    image = slo[0] if isinstance(slo, (tuple, list)) else slo
    return slo_encoder.body(image[:, None]).mean(dim=(2, 3)).float()


def pool(h, zparts=None, depth=None):
    if zparts is None:
        return h.mean(dim=(2, 3, 4))
    f = depth // h.shape[2]
    return torch.cat([h[:, :, lo // f: -(-hi // f)].mean(dim=(2, 3, 4)) for lo, hi in zparts], dim=1)


def encode(parts, base, x, timesteps, context=None, inject=None, zparts=None):
    emb = parts.time_embed(get_timestep_embedding(timesteps, base).to(dtype=x.dtype))
    if inject is not None:
        emb = inject.time(emb)
    h = parts.conv_in(x)
    if inject is not None:
        h = inject(0, h)
    skips, pooled = [h], []
    for i, block in enumerate(parts.down_blocks):
        h, res = block(hidden_states=h, temb=emb, context=context)
        if inject is not None and block.downsampler is not None:
            h = inject(i + 1, h)
            res[-1] = h
        skips.extend(res)
        pooled.append(pool(h, zparts, x.shape[2]))
    h = parts.middle_block(hidden_states=h, temb=emb, context=context)
    pooled.append(pool(h, zparts, x.shape[2]))
    return h, skips, emb, pooled


def decode(unet, h, skips, emb, context=None):
    for up in unet.up_blocks:
        n = len(up.resnets)
        res, skips = skips[-n:], skips[:-n]
        h = up(hidden_states=h, res_hidden_states_list=res, temb=emb, context=context)
    return unet.out(h)


class SloDiffusion(torch.nn.Module):
    def __init__(self, n_channels, adapter=False, z_down=Z_DOWN, tokens=8, dim=CONTEXT_DIM, xattn=()):
        super().__init__()
        self.unet = build_unet(n_channels, dim, z_down)
        self.slo = SloEncoder(dim=dim, tokens=tokens)
        ch = list(self.unet.block_out_channels)
        widths = [ch[0]] + ch[:-1]
        self.adapter = None
        if adapter:
            self.adapter = SloAdapter(self.slo.widths, widths, self.unet.time_embed[-1].out_features)
        self.xattn = None
        if xattn:
            if any(not 0 <= lv < len(widths) for lv in xattn):
                raise ValueError("--slo-xattn levels run from 0 to %d" % (len(widths) - 1))
            self.xattn = torch.nn.ModuleDict({str(lv): SloCrossAttention(widths[lv], dim) for lv in sorted(set(xattn))})

    def forward(self, x, timesteps, slo=None, keep=None):
        context, inject = condition(self.slo, self.adapter, slo, x.shape[0], keep, self.xattn)
        if inject is None:
            return self.unet(x=x, timesteps=timesteps, context=context)
        h, skips, emb, _ = encode(self.unet, self.unet.block_out_channels[0], x, timesteps, context, inject)
        return decode(self.unet, h, skips, emb, context)


def build_model(n_channels, slo_size=0, adapter=False, z_down=Z_DOWN, tokens=8, dim=CONTEXT_DIM, xattn=()):
    if (adapter or xattn) and not slo_size:
        raise ValueError("the SLO adapter and SLO cross-attention need an SLO size")
    if slo_size:
        return SloDiffusion(n_channels, adapter, z_down, tokens, dim, tuple(xattn))
    return build_unet(n_channels, None, z_down)


def predict(model, x, t, slo=None):
    if isinstance(model, SloDiffusion):
        return model(x, t, slo)
    return model(x=x, timesteps=t)


def to_device(cond, device):
    if cond is None:
        return None
    if isinstance(cond, (tuple, list)):
        return tuple(c.to(device) for c in cond)
    return cond.to(device)


class Encoder(torch.nn.Module):
    def __init__(self, model, slo_features=False, zparts=None):
        super().__init__()
        conditioned = isinstance(model, SloDiffusion)
        if slo_features and not conditioned:
            raise ValueError("SLO features need a model with an SLO encoder")
        unet = model.unet if conditioned else model
        self.slo = model.slo if conditioned else None
        self.adapter = model.adapter if conditioned else None
        self.xattn = getattr(model, "xattn", None) if conditioned else None
        self.slo_features = slo_features
        self.zparts = zparts
        self.base = unet.block_out_channels[0]
        self.time_embed = unet.time_embed
        self.conv_in = unet.conv_in
        self.down_blocks = unet.down_blocks
        self.middle_block = unet.middle_block
        self.dim = (sum(unet.block_out_channels) + unet.block_out_channels[-1]) * (len(zparts) if zparts else 1)
        if slo_features:
            self.dim += self.slo.widths[-1]

    def forward(self, x, t, slo=None):
        context, inject = None, None
        if self.slo is not None:
            context, inject = condition(self.slo, self.adapter, slo, x.shape[0], None, self.xattn)
        _, _, _, pooled = encode(self, self.base, x, t, context, inject, self.zparts)
        if self.slo_features:
            pooled.append(slo_vector(self.slo, slo, x.shape[0], x.device))
        return torch.cat([p.float() for p in pooled], dim=1)


def options(ckpt):
    return {"adapter": bool(ckpt.get("slo_adapter", False)), "keep": int(ckpt.get("keep_slices", 0)), "z_down": tuple(ckpt.get("z_down", Z_DOWN)),
            "tokens": int(ckpt.get("slo_tokens", 8)), "dim": int(ckpt.get("slo_dim", CONTEXT_DIM)), "xattn": tuple(ckpt.get("slo_xattn", ())), "slo_slice": ckpt.get("slo_slice")}


def model_args(ckpt):
    opt = options(ckpt)
    return dict(slo_size=ckpt.get("slo_size", 0), adapter=opt["adapter"], z_down=opt["z_down"], tokens=opt["tokens"], dim=opt["dim"], xattn=opt["xattn"])


def joint_parts(ckpt, axes):
    opt = options(ckpt)
    if not opt["slo_slice"]:
        return None
    zf = 2 ** sum(1 for a in axes if a == wv.Z)
    n = opt["keep"] // zf
    return ((0, n), (n, n + 1))


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    axes = wv.DESIGNS[ckpt["design"]]
    model = build_model(wv.channels(axes), **model_args(ckpt))
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt, axes


def band_scale(ckpt):
    band = ckpt.get("band_scale")
    return None if band is None else torch.tensor(band, dtype=torch.float32)
