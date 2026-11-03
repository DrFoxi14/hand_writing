from __future__ import annotations
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - exercised only without optional torch
    raise RuntimeError("Install the 'training' extra to use the PyTorch writer model") from exc

class StyleEncoder(nn.Module):
    def __init__(self, style_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.LeakyReLU(.2), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.LeakyReLU(.2), nn.MaxPool2d(2),
            nn.Conv2d(64,96,3,padding=1), nn.LeakyReLU(.2), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(96, style_dim), nn.Tanh(),
        )
    def forward(self, style): return self.net(style)

class WriterConditionedGenerator(nn.Module):
    """Character-sequence + reference-line conditioned grayscale handwriting generator."""
    def __init__(self, vocab_size: int, char_dim: int = 96, hidden_dim: int = 128, style_dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, char_dim, padding_idx=0)
        self.sequence = nn.GRU(char_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.style_encoder = StyleEncoder(style_dim)
        channels = hidden_dim + style_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(channels,128,3,padding=1), nn.GroupNorm(8,128), nn.LeakyReLU(.2),
            nn.Conv2d(128,96,3,padding=1), nn.GroupNorm(8,96), nn.LeakyReLU(.2),
            nn.Conv2d(96,64,3,padding=1), nn.GroupNorm(8,64), nn.LeakyReLU(.2),
            nn.Conv2d(64,1,3,padding=1), nn.Sigmoid(),
        )

    def forward(self, char_ids, style_image, output_height: int = 64, pixels_per_char: int = 20):
        seq, _ = self.sequence(self.embedding(char_ids))
        width = max(16, char_ids.shape[1] * pixels_per_char)
        seq_map = F.interpolate(seq.transpose(1,2), size=width, mode="linear", align_corners=False)
        seq_map = seq_map.unsqueeze(2).expand(-1,-1,output_height,-1)
        style = self.style_encoder(style_image).unsqueeze(-1).unsqueeze(-1)
        style_map = style.expand(-1,-1,output_height,width)
        return self.decoder(torch.cat([seq_map, style_map], dim=1))
