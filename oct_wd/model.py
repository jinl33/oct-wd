import torch
import torch.nn.functional as F
from generative.networks.nets import DiffusionModelUNet
from generative.networks.nets.diffusion_model_unet import get_timestep_embedding
from generative.networks.schedulers import DDPMScheduler

from oct_wd import wavelet as wv

CONTEXT_DIM = 160
Z_DOWN = (1, 1, 1, 1)
ALIGN_LEVEL = 2
ALIGN_STAGE = 2
ALIGN_DIM = 128
ALIGN_TEMP = 0.1
JEPA_MOMENTUM = 0.996


class ZUpsample(torch.nn.Module):
    def __init__(self, up):
        super().__init__()
        self.conv = up.conv

    def forward(self, x, emb=None):
        dtype = x.dtype
        x = F.interpolate(x.float() if dtype == torch.bfloat16 else x, scale_factor=(1.0, 2.0, 2.0), mode="nearest").to(dtype)
        return self.conv(x)


def build_unet(n_channels, context_dim=None, z_down=Z_DOWN, extra_in=0):
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=n_channels + extra_in,
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


def encode(parts, base, x, timesteps, context=None, inject=None, zparts=None, maps=None):
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
        if maps is not None:
            maps.append(h)
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


class SloStep(torch.nn.Module):
    def __init__(self, part, base, temb_dim):
        super().__init__()
        self.part = tuple(part)
        self.base = base
        self.mlp = torch.nn.Sequential(torch.nn.Linear(base, temb_dim), torch.nn.SiLU(), torch.nn.Linear(temb_dim, temb_dim))
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)
        std = (1 - DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta").alphas_cumprod).sqrt().float()
        self.register_buffer("std", std, persistent=False)

    def level(self, t):
        return self.std[t.clamp_min(0)] * (t >= 0)

    def inputs(self, x, t, t_slo=None):
        t_slo = t if t_slo is None else t_slo
        lo, hi = self.part
        m = self.level(t)[:, None, None, None, None].repeat(1, 1, *x.shape[2:])
        m[:, :, lo:hi] = self.level(t_slo)[:, None, None, None, None]
        return torch.cat([x, m.to(x.dtype)], dim=1), StepInjection(self, t_slo)


class StepInjection:
    def __init__(self, step, t_slo):
        self.step, self.t_slo = step, t_slo

    def time(self, emb):
        return emb + self.step.mlp(get_timestep_embedding(self.t_slo + 1, self.step.base)).to(emb.dtype)

    def __call__(self, level, h):
        return h


class PartNoiseUNet(torch.nn.Module):
    def __init__(self, n_channels, part, z_down=Z_DOWN):
        super().__init__()
        self.unet = build_unet(n_channels, None, z_down, extra_in=1)
        self.step = SloStep(part, self.unet.block_out_channels[0], self.unet.time_embed[-1].out_features)

    def forward(self, x, timesteps, t_slo=None):
        x, inject = self.step.inputs(x, timesteps, t_slo)
        h, skips, emb, _ = encode(self.unet, self.unet.block_out_channels[0], x, timesteps, None, inject)
        return decode(self.unet, h, skips, emb)


class AlignedDiffusion(torch.nn.Module):
    def __init__(self, n_channels, z_down=Z_DOWN, dim=ALIGN_DIM):
        super().__init__()
        self.unet = build_unet(n_channels, None, z_down)
        self.slo = SloEncoder()
        ch = list(self.unet.block_out_channels)
        self.oct_dense = torch.nn.Linear(ch[ALIGN_LEVEL - 1], dim)
        self.slo_dense = torch.nn.Linear(self.slo.widths[ALIGN_STAGE], dim)
        self.oct_whole = torch.nn.Linear(sum(ch) + ch[-1], dim)
        self.slo_whole = torch.nn.Linear(self.slo.widths[-1], dim)

    def forward(self, x, timesteps, slo=None):
        return self.unet(x=x, timesteps=timesteps)

    def embed(self, x0, slo):
        if not isinstance(slo, (tuple, list)) or len(slo) != 2:
            raise ValueError("the aligned model needs the SLO together with its scan-line grid")
        image, grid = slo
        maps = []
        t0 = torch.zeros(x0.shape[0], dtype=torch.long, device=x0.device)
        _, _, _, pooled = encode(self.unet, self.unet.block_out_channels[0], x0, t0, maps=maps)
        h = maps[ALIGN_LEVEL - 1].mean(dim=3)
        stages = self.slo.stages(image)
        f = stages[ALIGN_STAGE]
        xy = grid[:, :2].permute(0, 2, 3, 1).contiguous()
        f = F.adaptive_avg_pool2d(F.grid_sample(f, xy.to(f.dtype), mode="bilinear", padding_mode="zeros", align_corners=False), h.shape[2:])
        valid = F.adaptive_avg_pool2d(grid[:, 2:3].float(), h.shape[2:])[:, 0].flatten(1) > 0.999
        o = self.oct_dense(h.flatten(2).transpose(1, 2))
        v = self.slo_dense(f.flatten(2).transpose(1, 2))
        go = self.oct_whole(torch.cat([p.float() for p in pooled], dim=1))
        gs = self.slo_whole(stages[-1].mean(dim=(2, 3)))
        return tuple(F.normalize(a.float(), dim=-1) for a in (o, v)) + (valid,) + tuple(F.normalize(a.float(), dim=-1) for a in (go, gs))

    def match(self, x0, slo):
        return self.embed(x0, slo)


def predictor(dim):
    return torch.nn.Sequential(torch.nn.Linear(dim, 2 * dim), torch.nn.GELU(), torch.nn.Linear(2 * dim, dim))


def latent_distance(p, z):
    return 2 - 2 * (F.normalize(p.float(), dim=-1) * z).sum(-1)


class JepaAligned(AlignedDiffusion):
    def __init__(self, n_channels, z_down=Z_DOWN, dim=ALIGN_DIM):
        super().__init__(n_channels, z_down, dim)
        target = AlignedDiffusion(n_channels, z_down, dim)
        target.load_state_dict(self.state_dict())
        for p in target.parameters():
            p.requires_grad_(False)
        self.oct_to_slo = predictor(dim)
        self.slo_to_oct = predictor(dim)
        self.whole_oct_to_slo = predictor(dim)
        self.whole_slo_to_oct = predictor(dim)
        self.target = target

    def terms(self, x0, slo):
        o, v, valid, go, gs = self.embed(x0, slo)
        with torch.no_grad():
            ot, vt, _, got, gst = self.target.embed(x0, slo)
        dense = (latent_distance(self.oct_to_slo(o), vt)[valid].mean() + latent_distance(self.slo_to_oct(v), ot)[valid].mean()) / 2
        whole = (latent_distance(self.whole_oct_to_slo(go), gst).mean() + latent_distance(self.whole_slo_to_oct(gs), got).mean()) / 2
        return dense, whole

    def match(self, x0, slo):
        o, _, valid, go, _ = self.embed(x0, slo)
        _, vt, _, _, gst = self.target.embed(x0, slo)
        return F.normalize(self.oct_to_slo(o).float(), dim=-1), vt, valid, F.normalize(self.whole_oct_to_slo(go).float(), dim=-1), gst

    @torch.no_grad()
    def update_target(self, momentum=JEPA_MOMENTUM):
        online = dict(self.named_parameters())
        for name, p in self.target.named_parameters():
            p.mul_(momentum).add_(online[name].detach().to(p.dtype), alpha=1 - momentum)


def info_nce(a, b, temp=ALIGN_TEMP):
    logits = a @ b.t() / temp
    target = torch.arange(a.shape[0], device=a.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target)) / 2


def align_loss(o, v, valid, go, gs, temp=ALIGN_TEMP):
    m = valid.reshape(-1)
    dense = info_nce(o.reshape(-1, o.shape[-1])[m], v.reshape(-1, v.shape[-1])[m], temp)
    whole = info_nce(go, gs, temp) if go.shape[0] > 1 else dense.new_zeros(())
    return dense, whole


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


def build_model(n_channels, slo_size=0, adapter=False, z_down=Z_DOWN, tokens=8, dim=CONTEXT_DIM, xattn=(), part=None, align=False, align_loss="contrastive"):
    if (adapter or xattn) and not slo_size:
        raise ValueError("the SLO adapter and SLO cross-attention need an SLO size")
    if align:
        if not slo_size or adapter or xattn or part is not None:
            raise ValueError("the aligned model needs an SLO size and no SLO adapter, cross-attention or SLO slice")
        if align_loss not in ("contrastive", "jepa"):
            raise ValueError("unknown alignment loss %s" % align_loss)
        return JepaAligned(n_channels, z_down) if align_loss == "jepa" else AlignedDiffusion(n_channels, z_down)
    if part is not None:
        if slo_size:
            raise ValueError("an own noise step for the SLO slice needs the joint model, without an SLO encoder")
        return PartNoiseUNet(n_channels, part, z_down)
    if slo_size:
        return SloDiffusion(n_channels, adapter, z_down, tokens, dim, tuple(xattn))
    return build_unet(n_channels, None, z_down)


def predict(model, x, t, slo=None, t_slo=None):
    if isinstance(model, SloDiffusion):
        return model(x, t, slo)
    if isinstance(model, PartNoiseUNet):
        return model(x, t, t_slo)
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
        aligned = isinstance(model, AlignedDiffusion)
        if slo_features and not (conditioned or aligned):
            raise ValueError("SLO features need a model with an SLO encoder")
        self.step = model.step if isinstance(model, PartNoiseUNet) else None
        self.conditions = conditioned
        unet = model.unet if conditioned or aligned or self.step is not None else model
        self.slo = model.slo if conditioned or aligned else None
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
        if self.conditions:
            context, inject = condition(self.slo, self.adapter, slo, x.shape[0], None, self.xattn)
        if self.step is not None:
            x, inject = self.step.inputs(x, t)
        _, _, _, pooled = encode(self, self.base, x, t, context, inject, self.zparts)
        if self.slo_features:
            pooled.append(slo_vector(self.slo, slo, x.shape[0], x.device))
        return torch.cat([p.float() for p in pooled], dim=1)


def options(ckpt):
    return {"adapter": bool(ckpt.get("slo_adapter", False)), "keep": int(ckpt.get("keep_slices", 0)), "z_down": tuple(ckpt.get("z_down", Z_DOWN)),
            "tokens": int(ckpt.get("slo_tokens", 8)), "dim": int(ckpt.get("slo_dim", CONTEXT_DIM)), "xattn": tuple(ckpt.get("slo_xattn", ())), "slo_slice": ckpt.get("slo_slice"),
            "own_step": bool(ckpt.get("slo_own_step", False)), "align": float(ckpt.get("slo_align", 0.0)), "align_loss": ckpt.get("slo_align_loss", "contrastive")}


def model_args(ckpt):
    opt = options(ckpt)
    part = joint_parts(ckpt, wv.DESIGNS[ckpt["design"]])[1] if opt["own_step"] else None
    return dict(slo_size=ckpt.get("slo_size", 0), adapter=opt["adapter"], z_down=opt["z_down"], tokens=opt["tokens"], dim=opt["dim"], xattn=opt["xattn"], part=part, align=opt["align"] > 0, align_loss=opt["align_loss"])


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
