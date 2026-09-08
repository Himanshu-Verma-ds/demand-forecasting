from __future__ import annotations

import torch
from torch import nn


class GlobalLSTMForecaster(nn.Module):
    """Global encoder-decoder LSTM with static embeddings and known-future inputs.

    Shape journey (B=batch, L=lookback, H=14):
      past_x    [B,L,P] -> encoder LSTM -> h,c [layers,B,hidden]
      static    [B,S]   -> embeddings -> static_vec [B,S*emb]
      future_x  [B,H,F]
      decoder input each step = concat(future_x[:,h,:], static_vec, previous_y)
      outputs   [B,H]

    The same weights are shared across every store-SKU series. Series identity and hierarchy
    enter through embeddings, so the model can learn global patterns while retaining heterogeneity.
    """
    def __init__(self, past_dim: int, future_dim: int, cardinalities: list[int], hidden: int = 128,
                 emb_dim: int = 16, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cardinalities])
        static_dim = len(cardinalities) * emb_dim
        self.encoder = nn.LSTM(past_dim, hidden, num_layers=num_layers, batch_first=True,
                               dropout=dropout if num_layers > 1 else 0.0)
        self.decoder = nn.LSTM(future_dim + static_dim + 1, hidden, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def forward(self, past_x, future_x, static_ids, teacher_y=None, teacher_forcing: float = 0.0):
        B, H, _ = future_x.shape
        static_vec = torch.cat([emb(static_ids[:, i]) for i, emb in enumerate(self.embeddings)], dim=-1)
        _, state = self.encoder(past_x)
        # Last normalized demand in channel 0 is the first autoregressive token.
        prev_y = past_x[:, -1, 0:1]
        preds = []
        for h in range(H):
            dec_in = torch.cat([future_x[:, h, :], static_vec, prev_y], dim=-1).unsqueeze(1)
            dec_out, state = self.decoder(dec_in, state)
            yhat = self.head(dec_out[:, 0, :])  # [B,1]
            preds.append(yhat)
            if self.training and teacher_y is not None and teacher_forcing > 0:
                use_teacher = (torch.rand(B, 1, device=yhat.device) < teacher_forcing).float()
                prev_y = use_teacher * teacher_y[:, h:h+1] + (1 - use_teacher) * yhat
            else:
                prev_y = yhat
        return torch.cat(preds, dim=1)
