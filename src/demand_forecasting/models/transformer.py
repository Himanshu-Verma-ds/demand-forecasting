from __future__ import annotations

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GlobalTemporalTransformer(nn.Module):
    """Transformer forecaster from scratch with direct 14-step output.

    past_x [B,L,P] -> linear projection [B,L,D] -> TransformerEncoder [B,L,D]
    future_x [B,H,F] -> projection [B,H,D]
    static embeddings [B,E] -> projection [B,1,D]
    Future queries = future projection + static context; cross-attend to encoded history.
    output [B,H,1] -> [B,H].

    Direct decoding means each horizon step is predicted simultaneously, avoiding recursive error
    propagation while still using known future promo/calendar/price/weather for each future day.
    """
    def __init__(self, past_dim: int, future_dim: int, cardinalities: list[int], d_model: int = 128,
                 nhead: int = 4, num_layers: int = 3, emb_dim: int = 16, dropout: float = 0.15):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cardinalities])
        static_dim = len(cardinalities) * emb_dim
        self.past_proj = nn.Linear(past_dim, d_model)
        self.future_proj = nn.Linear(future_dim, d_model)
        self.static_proj = nn.Linear(static_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=4*d_model,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=max(1, num_layers - 1))
        self.head = nn.Linear(d_model, 1)

    def forward(self, past_x, future_x, static_ids):
        static_vec = torch.cat([emb(static_ids[:, i]) for i, emb in enumerate(self.embeddings)], dim=-1)
        memory = self.encoder(self.pos(self.past_proj(past_x)))                     # [B,L,D]
        queries = self.pos(self.future_proj(future_x) + self.static_proj(static_vec).unsqueeze(1))
        decoded = self.decoder(tgt=queries, memory=memory)                          # [B,H,D]
        return self.head(decoded).squeeze(-1)                                      # [B,H]
