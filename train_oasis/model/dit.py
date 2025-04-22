"""
References:
    - DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/unet3d.py
    - Latte: https://github.com/Vchitect/Latte/blob/main/models/latte.py
"""

from typing import Optional
import torch
from torch import nn
from train_oasis.model.rotary_embedding_torch import RotaryEmbedding
from einops import rearrange
from train_oasis.model.attention import SpatialAxialAttention, TemporalAxialAttention
from timm.models.vision_transformer import Mlp
from .blocks import (
    PatchEmbed, 
    modulate, 
    gate,
    FinalLayer,
    TimestepEmbedder,
)
from torch.utils.checkpoint import checkpoint

class SpatioTemporalDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        is_causal=True,
        spatial_rotary_emb: Optional[RotaryEmbedding] = None,
        temporal_rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.is_causal = is_causal
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.s_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_attn = SpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=spatial_rotary_emb,
        )
        self.s_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.s_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.t_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_attn = TemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            is_causal=is_causal,
            rotary_emb=temporal_rotary_emb,
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.t_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        B, T, H, W, D = x.shape

        # spatial block
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)

        # x_norm_spatial = modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)
        # x_spatial = self.s_attn(x_norm_spatial, kv_override=getattr(self, 'spatial_kv_override', None))
        # x = x + gate(x_spatial, s_gate_msa)


        x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)


        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.t_attn(modulate(self.t_norm1(x), t_shift_msa, t_scale_msa)), t_gate_msa)
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_h=18,
        input_w=32,
        patch_size=2,
        in_channels=16,
        hidden_size=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        external_cond_dim=25,
        max_frames=32,
        gradient_checkpointing=False,
        dtype=torch.float32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.max_frames = max_frames
        self.gradient_checkpointing = gradient_checkpointing
        self.dtype = dtype

        self.x_embedder = PatchEmbed(input_h, input_w, patch_size, in_channels, hidden_size, flatten=False)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        frame_h, frame_w = self.x_embedder.grid_size

        self.spatial_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads // 2, freqs_for="pixel", max_freq=256)
        self.temporal_rotary_emb = RotaryEmbedding(dim=hidden_size // num_heads)
        self.external_cond = nn.Linear(external_cond_dim, hidden_size) if external_cond_dim > 0 else nn.Identity()

        self.blocks = nn.ModuleList(
            [
                SpatioTemporalDiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=True,
                    spatial_rotary_emb=self.spatial_rotary_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, H, W, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = x.shape[1]
        w = x.shape[2]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, external_cond=None):
        """
        Forward pass of DiT.
        x: (B, T, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (B, T,) tensor of diffusion timesteps
        """

        B, T, C, H, W = x.shape

        # add spatial embeddings
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.x_embedder(x)  # (B*T, C, H, W) -> (B*T, H/2, W/2, D) , C = 16, D = d_model
        # restore shape
        x = rearrange(x, "(b t) h w d -> b t h w d", t=T)
        # embed noise steps
        t = rearrange(t, "b t -> (b t)")
        c = self.t_embedder(t)  # (N, D)
        # print(f"c shape: {c.shape}, x.shape: {x.shape}")

        c = rearrange(c, "(b t) d -> b t d", t=T)
        # assert c.shape[0] == B * T, f"Expected c.shape[0]={B*T}, got {c.shape[0]}"
        # T = c.shape[0] // B
        # c = rearrange(c, "(b t) d -> b t d", b=B, t=T)
        if torch.is_tensor(external_cond):
            c += self.external_cond(external_cond)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)  # (N, T, H, W, D)
        if self.gradient_checkpointing and self.training:
            x = checkpoint(self.final_layer, x, c, use_reentrant=False)
        else:
            x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", t=T)

        return x
    
    def inject_spatial_kv(self, ks, vs):
        num_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            if hasattr(block, "s_attn") and hasattr(block.s_attn, "set_kv_override"):
                if i >= num_blocks // 2:
                    block.s_attn.set_kv_override(ks[i], vs[i])
        # num_blocks = len(self.blocks)
        # block = self.blocks[num_blocks - 1]
        # if hasattr(block, "s_attn") and hasattr(block.s_attn, "set_kv_override"):
        #     block.s_attn.set_kv_override(k, v)

    def clear_spatial_kv(self):
        for block in self.blocks:
            if hasattr(block, "s_attn") and hasattr(block.s_attn, "clear_kv_override"):
                block.s_attn.clear_kv_override()


def DiT_S_2():
    return DiT(
        patch_size=2,
        hidden_size=1024,
        depth=16,
        num_heads=16,
    )

def dit_small():
    return DiT(
        input_h=64,
        input_w=64,
        in_channels=3,
        patch_size=2,
        hidden_size=256,
        depth=16,
        num_heads=16,
        external_cond_dim=4,
        max_frames=10
    )

def dit_cty():
    return DiT(
        input_h=18,
        input_w=32,
        in_channels=16,
        patch_size=2,
        hidden_size=1024,
        depth=16,
        num_heads=16,
        external_cond_dim=25,
        max_frames=10
    )

def flappy_bird_dit():
    return DiT(
        input_h=64,
        input_w=36,
        in_channels=4,
        patch_size=2,
        hidden_size=512,
        depth=8,
        num_heads=16,
        external_cond_dim=2,
        max_frames=10
    )

def flappy_bird_dit_half():
    return DiT(
        input_h=32,
        input_w=18,
        in_channels=4,
        patch_size=2,
        hidden_size=512,
        depth=8,
        num_heads=16,
        external_cond_dim=2,
        max_frames=10
    )

DiT_models = {
    "DiT-S/2": DiT_S_2,
    "dit_small": dit_small,
    "dit_cty": dit_cty,
    "flappy_bird_dit": flappy_bird_dit,
    "flappy_bird_dit_half": flappy_bird_dit_half,
}
