import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from time_schedules import create_all_registered_time_sequences


class CausalConv1dSame(nn.Module):
    """
    Causal Conv1d with stride=1, dilation=1, and output length == input length.

    Input:  (N, Cin, L)
    Output: (N, Cout, L)
    """

    def __init__(self, cin: int, cout: int, kernel_size: int, bias: bool = True, **conv_kwargs):
        super().__init__()
        assert kernel_size >= 1
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            cin,
            cout,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            padding=0,
            bias=bias,
            **conv_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class FFBlock(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout=0):
        super(FFBlock, self).__init__()
        self.L0 = nn.Linear(embedding_size, hidden_size)
        self.L1 = nn.Linear(hidden_size, embedding_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(embedding_size)

    def forward(self, x):
        h = self.layernorm(x)
        h = self.dropout(h)
        h = self.L0(h)
        h = self.act(h)
        h = self.L1(h)
        x = x + h
        return x


class AttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads=8, dropout=0):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout / 4,
        )

    def forward(self, x):
        h = self.layer_norm(x)
        h = self.dropout(h)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.attention(h, h, h, attn_mask=mask, is_causal=True)[0]
        x = x + h
        return x


class FeatureEncoder(nn.Module):
    def __init__(self, num_features, embedding_dim, conv_kernel_size=16):
        super(FeatureEncoder, self).__init__()
        self.linear = nn.Linear(num_features, embedding_dim)
        self.conv = CausalConv1dSame(1, embedding_dim, kernel_size=conv_kernel_size)

    def forward(self, static_features, time_series):
        # This is for scalar time series, so time_series has shape B x T.
        feature_mapping = self.linear(static_features)
        feature_mapping = feature_mapping.unsqueeze(1).repeat([1, time_series.size(1), 1])
        embedded_time_series = feature_mapping + torch.permute(
            self.conv(time_series.unsqueeze(1)), [0, 2, 1]
        )
        return embedded_time_series


class PositionalEncoding(nn.Module):
    """
    Fourier positional encoding for sampled HGR time series.

    Backward-compatible behavior:
        t can still be a batch of integer schedule ids, shape [B].

    New flexible behavior:
        t can also be explicit time values, shape [T] or [B, T]. This means the
        loader/model can handle any regular periodic time grid whose actual time
        columns are present in the CSV header, without turning the schedule id
        into a boolean.
    """

    def __init__(self, embedding_size, MAX_LEN=256, num_fourier_channels=None):
        super(PositionalEncoding, self).__init__()
        if num_fourier_channels is None:
            num_fourier_channels = embedding_size // 2
        self.embedding_size = embedding_size
        self.num_fourier_channels = num_fourier_channels

        # Registered schedule-id path. This preserves the old fast lookup path.
        time_seqs = create_all_registered_time_sequences(MAX_LEN)
        fourier_seqs = self._create_fourier_from_times(time_seqs, num_fourier_channels)
        self.register_buffer("fourier_seqs", fourier_seqs)

    def forward(self, embedding, t):
        # embedding is the embedding of the time series: [B, T, E]
        batch_size = embedding.size(0)
        seq_len = embedding.size(1)
        device = embedding.device

        t = torch.as_tensor(t, device=device)

        if self._looks_like_schedule_id_batch(t, batch_size):
            schedule_ids = t.long().view(-1)
            if schedule_ids.min().item() < 0 or schedule_ids.max().item() >= self.fourier_seqs.size(0):
                raise ValueError(
                    f"Got schedule ids in [{schedule_ids.min().item()}, {schedule_ids.max().item()}], "
                    f"but PositionalEncoding only has registered ids 0..{self.fourier_seqs.size(0) - 1}. "
                    "Either add the schedule to TIME_SCHEDULE_GAPS or pass explicit time values."
                )
            fourier = self.fourier_seqs.index_select(0, schedule_ids)[:, :seq_len, :]
        else:
            # Explicit time-value path.
            # Accept [T] and broadcast to [B, T], or accept [B, T] directly.
            time_values = t.float()
            if time_values.ndim == 1:
                if time_values.numel() < seq_len:
                    raise ValueError(
                        f"Time vector length {time_values.numel()} is shorter than sequence length {seq_len}."
                    )
                time_values = time_values[:seq_len].unsqueeze(0).repeat(batch_size, 1)
            elif time_values.ndim == 2:
                if time_values.size(0) == 1:
                    time_values = time_values.repeat(batch_size, 1)
                elif time_values.size(0) != batch_size:
                    raise ValueError(
                        f"Time value batch has size {time_values.size(0)}, expected 1 or {batch_size}."
                    )
                if time_values.size(1) < seq_len:
                    raise ValueError(
                        f"Time vector length {time_values.size(1)} is shorter than sequence length {seq_len}."
                    )
                time_values = time_values[:, :seq_len]
            else:
                raise ValueError(f"t must be schedule ids [B], time values [T], or time values [B,T]. Got {tuple(t.shape)}")

            fourier = self._create_fourier_from_times(time_values, self.num_fourier_channels)

        if fourier.size(-1) < embedding.size(-1):
            padding = torch.zeros(
                fourier.size(0),
                fourier.size(1),
                embedding.size(2) - fourier.size(2),
                device=embedding.device,
                dtype=embedding.dtype,
            )
            fourier = torch.cat((fourier.to(dtype=embedding.dtype), padding), -1)
        else:
            fourier = fourier[..., : embedding.size(-1)].to(dtype=embedding.dtype)

        return embedding + fourier

    @staticmethod
    def _looks_like_schedule_id_batch(t: torch.Tensor, batch_size: int) -> bool:
        # Old code passed t as one scalar-ish value per row. That is [B].
        # Explicit time values are usually floats and either [T] or [B, T].
        if t.ndim == 0:
            return batch_size == 1
        if t.ndim != 1:
            return False
        if t.numel() != batch_size:
            return False
        # Integer/bool tensors are definitely ids. Float tensors of length B are
        # also treated as ids for backwards compatibility with old datasets.
        return True

    @staticmethod
    def _create_time_seqs(MAX_LEN: int):
        """
        Backward-compatible helper for old code that calls
        PositionalEncoding._create_time_seqs(MAX_LEN).
        """
        stacked = create_all_registered_time_sequences(MAX_LEN)
        return [stacked[i] for i in range(stacked.size(0))]

    @staticmethod
    def _create_fourier_seq(time_seq, num_fourier_channels: int):
        return PositionalEncoding._create_fourier_from_times(
            torch.as_tensor(time_seq, dtype=torch.float32).unsqueeze(0),
            num_fourier_channels,
        ).squeeze(0)

    @staticmethod
    def _create_fourier_from_times(time_values, num_fourier_channels: int):
        """
        Vectorized Fourier features.

        Args:
            time_values: [T], [B,T], or [num_schedules,T]
            num_fourier_channels: number of Fourier channels to create

        Returns:
            Tensor with shape time_values.shape + [num_fourier_channels]
        """
        times = torch.as_tensor(time_values, dtype=torch.float32)
        if times.ndim == 1:
            times = times.unsqueeze(0)

        device = times.device
        fourier = torch.zeros(*times.shape, num_fourier_channels, device=device, dtype=times.dtype)
        n_pairs = num_fourier_channels // 2
        if n_pairs == 0:
            return fourier

        channels = torch.arange(n_pairs, device=device, dtype=times.dtype)
        denom = torch.pow(torch.tensor(10000.0, device=device, dtype=times.dtype), 2 * channels / num_fourier_channels)
        angles = times.unsqueeze(-1) / denom.view(*([1] * times.ndim), n_pairs)
        fourier[..., 0 : 2 * n_pairs : 2] = torch.sin(angles)
        fourier[..., 1 : 2 * n_pairs : 2] = torch.cos(angles)
        return fourier


class StaticFeatureTransformer(nn.Module):
    def __init__(self, num_features, embedding_size, num_layers, hidden_size, num_heads, dropout=0.35):
        super(StaticFeatureTransformer, self).__init__()
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.positional_encoding = PositionalEncoding(embedding_size)
        self.feature_encoder = FeatureEncoder(num_features, embedding_size)
        layer_list = nn.ModuleList([])
        for _ in range(num_layers):
            layer_list.append(FFBlock(embedding_size, hidden_size, dropout=dropout))
            layer_list.append(AttentionBlock(embedding_size, num_heads, dropout=dropout))
        self.layers = layer_list
        self.output_dropout = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(embedding_size, 1)

    def forward(self, x, t, y):
        h = self.feature_encoder(x, y)
        h = self.positional_encoding(h, t)
        for layer in self.layers:
            h = layer(h)
        return self.output_layer(h)

    def decode(self, x, t, T):
        y = torch.zeros(x.size(0), 1, 1, device=x.device)
        for _ in range(T):
            y_hat = self.forward(x, t, torch.flatten(y, 1, 2))
            y = torch.cat([y, y_hat[:, -1].unsqueeze(-1)], 1)
        return y


class ConvBlock(nn.Module):
    def __init__(self, embed_dim, kernel_size, dropout_p=0.5):
        super().__init__()
        self.conv0 = CausalConv1dSame(embed_dim, embed_dim, kernel_size)
        self.conv1 = CausalConv1dSame(embed_dim, embed_dim, kernel_size)
        self.dropout = nn.Dropout(dropout_p)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.dropout(self.norm(x))
        h = torch.transpose(h, 1, 2).contiguous()
        h = self.conv0(h)
        h = self.act(h)
        h = self.conv1(h)
        h = torch.transpose(h, 1, 2).contiguous()
        return x + h


class StaticFeatureTCN(nn.Module):
    def __init__(self, num_features, embedding_size, num_layers, kernel_size, dropout=0.35):
        super(StaticFeatureTCN, self).__init__()
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.positional_encoding = PositionalEncoding(embedding_size)
        self.feature_encoder = FeatureEncoder(num_features, embedding_size)
        layer_list = nn.ModuleList([])
        for _ in range(num_layers):
            layer_list.append(ConvBlock(embedding_size, kernel_size))
        self.layers = layer_list
        self.output_dropout = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(embedding_size, 1)

    def forward(self, x, t, y):
        h = self.feature_encoder(x, y)
        h = self.positional_encoding(h, t)
        for layer in self.layers:
            h = layer(h)
        return self.output_layer(self.output_dropout(h))

    def decode(self, x, t, T):
        y = torch.zeros(x.size(0), 1, device=x.device)
        for _ in range(1, T):
            y_hat = self.forward(x, t, y)
            y = torch.cat([y, y_hat[:, -1]], 1)
        return y


if __name__ == "__main__":
    # Quick smoke tests for both accepted t formats.
    x = torch.randn(64, 14)
    y = torch.randn(64, 72)
    model = StaticFeatureTCN(14, 256, 2, 8, dropout=0)

    schedule_ids = torch.zeros(64, dtype=torch.long)
    out = model(x, schedule_ids, y)
    print("schedule id path:", out.shape)

    explicit_times = PositionalEncoding._create_time_seqs(72)[0].unsqueeze(0).repeat(64, 1)
    out = model(x, explicit_times, y)
    print("explicit time-value path:", out.shape)
