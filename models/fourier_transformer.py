import torch
import torch.nn as nn
import torch.nn.functional as F


class GeGELU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim * 2),
            GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FourierLinearAttention(nn.Module):
    """
    Fourier type-linear attention from Shih et al. (2025):
        Attention(Q, K, V) = Q_hat (K_hat^T V) / n
    where Q_hat and K_hat are layer-normalized per head.
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.embed_dim)

    def forward(self, query, context=None):
        if context is None:
            context = query

        q = self.q_norm(self._split_heads(self.q_proj(query)))
        k = self.k_norm(self._split_heads(self.k_proj(context)))
        v = self._split_heads(self.v_proj(context))

        n = k.shape[-2]
        kv = torch.einsum("bhnd,bhne->bhde", k, v) / n
        x = torch.einsum("bhmd,bhde->bhme", q, kv)
        return self.out_proj(self._merge_heads(x))


class FourierEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = FourierLinearAttention(embed_dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, embed_dim * mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class FourierDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        self.self_attn = FourierLinearAttention(embed_dim, num_heads, dropout)
        self.cross_query_norm = nn.LayerNorm(embed_dim)
        self.cross_context_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = FourierLinearAttention(embed_dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, embed_dim * mlp_ratio, dropout)

    def forward(self, x, context):
        x = x + self.self_attn(self.self_attn_norm(x))
        x = x + self.cross_attn(
            self.cross_query_norm(x),
            self.cross_context_norm(context),
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x


class FourierTransformer1D(nn.Module):
    def __init__(
        self,
        in_channel=1,
        out_channel=1,
        embed_dim=96,
        num_heads=4,
        encoder_layers=4,
        decoder_layers=3,
        mlp_ratio=4,
        dropout=0.0,
    ):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.embed_dim = embed_dim

        self.input_embedding = nn.Sequential(
            nn.Linear(in_channel + 1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.query_embedding = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.encoder = nn.ModuleList(
            [
                FourierEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(encoder_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                FourierDecoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(decoder_layers)
            ]
        )

        self.output_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * mlp_ratio * 2),
            GeGELU(),
            nn.Linear(embed_dim * mlp_ratio, out_channel),
        )

    def forward(self, x, grid):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len) when in_channel == 1,
                or (batch_size, seq_len, in_channel).
            grid: Tensor of shape (batch_size, seq_len).

        Returns:
            Tensor of shape (batch_size, seq_len) when out_channel == 1.
        """
        if x.ndim == 2:
            if self.in_channel != 1:
                raise ValueError(
                    "2D x is only valid when in_channel == 1; "
                    f"got in_channel == {self.in_channel}"
                )
            x = x.unsqueeze(-1)
        elif x.ndim != 3:
            raise ValueError(
                "x must have shape (batch_size, seq_len) or "
                "(batch_size, seq_len, in_channel)"
            )

        if x.shape[-1] != self.in_channel:
            raise ValueError(
                f"expected x.shape[-1] == {self.in_channel}, got {x.shape[-1]}"
            )
        if grid.ndim == 2:
            grid = grid.unsqueeze(-1)
        elif grid.ndim != 3 or grid.shape[-1] != 1:
            raise ValueError(
                "grid must have shape (batch_size, seq_len) or "
                "(batch_size, seq_len, 1)"
            )
        if x.shape[:2] != grid.shape[:2]:
            raise ValueError(
                f"x and grid must have matching batch and sequence dimensions; "
                f"got {x.shape[:2]} and {grid.shape[:2]}"
            )

        grid = grid.to(device=x.device, dtype=x.dtype)

        context = self.input_embedding(torch.cat((x, grid), dim=-1))
        query = self.query_embedding(grid)

        for block in self.encoder:
            context = block(context)
        for block in self.decoder:
            query = block(query, context)

        x = self.output_projection(query)
        if self.out_channel == 1:
            return x.squeeze(-1)
        return x


def test_fourier_transformer():
    batch_size = 2
    seq_len = 16
    model = FourierTransformer1D(
        embed_dim=8,
        num_heads=2,
        encoder_layers=1,
        decoder_layers=1,
        mlp_ratio=2,
    )
    x = torch.randn(batch_size, seq_len)
    grid = torch.linspace(0.0, 1.0, seq_len).expand(batch_size, -1)

    output = model(x, grid)

    assert output.shape == (batch_size, seq_len)
    print(output.shape)
if __name__ == "__main__":
    test_fourier_transformer()