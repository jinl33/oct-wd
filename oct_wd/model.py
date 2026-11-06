from generative.networks.nets import DiffusionModelUNet


def build_unet(n_channels):
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
        with_conditioning=False,
    )
