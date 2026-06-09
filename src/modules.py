import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights_for_relu(module: nn.Module, relu_slope=0.0):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_uniform_(module.weight, a=relu_slope, nonlinearity='leaky_relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=4, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)          # [B, C, H', W']
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    def __init__(self, embed_dim=384, img_size=32, patch_size=4, num_heads=6, depth=12, dropout=0.):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        num_patches = (img_size // patch_size) ** 2  # num_patches (noted L)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)  # shape [1, 1, E=embed_dim]
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)  # shape [1, 1 + L, E]
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)  # shape [B, L, E]
        cls = self.cls_token.expand(x.size(0), -1, -1)  # shape [B, 1, E]
        x = torch.cat([cls, x], dim=1)  # shape [B, 1 + L, E]
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.encoder(x)  # shape [B, 1 + L, E]
        return self.norm(x[:, 0])  # shape [B, E]


class DINOHead(nn.Module):
    def __init__(self, in_dim=384, hidden_dim=2048, out_dim=256, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.apply(self._init_weights)
        self.out = nn.Linear(hidden_dim, out_dim, bias=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init_weights_for_relu(m.weight)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.out(x)

    
class DINOModel(nn.Module):
    def __init__(self, embed_dim=384, img_size=32, patch_size=4, num_heads=6, depth=12, dropout=0., head_hidden_dim=2048, head_out_dim=256):
        super().__init__()
        self.backbone = ViT(embed_dim, img_size, patch_size, num_heads, depth, dropout)
        self.head = DINOHead(in_dim=embed_dim, hidden_dim=head_hidden_dim, out_dim=head_out_dim)

    def forward(self, x):
        return self.head(self.backbone(x))
