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
    
    def get_last_selfattention(self, x):
        x = self.patch_embed(x)  # shape [B, L, E]
        cls = self.cls_token.expand(x.size(0), -1, -1)  # shape [B, 1, E]
        x = torch.cat([cls, x], dim=1)  # shape [B, 1 + L, E]
        x = x + self.pos_embed
        for i, blk in enumerate(self.encoder):
            if i < len(self.encoder) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)


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
        self.out = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init_weights_for_relu(m.weight)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.out(x)


class MultiCropWrapper(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        # Group by input spatial size (height/width)
        # Use shape[-1] for square images
        sizes = torch.tensor([inp.shape[-1] for inp in x])
        unique_sizes, counts = torch.unique_consecutive(sizes, return_counts=True)
        idx_crops = torch.cumsum(counts, 0)
        start_idx = 0
        outputs = []
        for end_idx in idx_crops:
            # Concatenate all crops of this size
            group = torch.cat(x[start_idx:end_idx])
            out = self.backbone(group)   # output shape [B_group, embed_dim]
            if isinstance(out, tuple):
                out = out[0]              # for compatibility with XCiT
            outputs.append(out)
            start_idx = end_idx
        # Concatenate outputs from all groups and pass through head
        output = torch.cat(outputs)
        return self.head(output)
    
    
class DINOModel(nn.Module):
    def __init__(self, embed_dim=384, img_size=32, patch_size=4, num_heads=6, depth=12, dropout=0., head_hidden_dim=2048, head_out_dim=256):
        super().__init__()
        self.backbone = ViT(embed_dim, img_size, patch_size, num_heads, depth, dropout)
        self.head = DINOHead(in_dim=embed_dim, hidden_dim=head_hidden_dim, out_dim=head_out_dim)
        self.model = MultiCropWrapper(self.backbone, self.head)

    def forward(self, x):
        return self.model(x)


def test():
    crops = 2 + 6
    bs = 8
    img_size = 32
    embed_dim = 384
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = [torch.rand([bs, 3, img_size, img_size]).to(device) for _ in range(crops)]
    model = DINOModel(embed_dim, img_size, patch_size=4, num_heads=6, depth=12, dropout=0.,
                        head_hidden_dim=1024, head_out_dim=1024).to(device)
    out = model(data).detach()
    print(out)
    print(out.shape)

if __name__ == "__main__":
    test()
    