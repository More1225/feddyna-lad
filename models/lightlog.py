import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=out_channels)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out = out + residual
        out = self.relu(out)
        return out


class TCNBackbone(nn.Module):
    def __init__(
        self,
        input_dim=300,
        num_classes=2,
        filters_list=None,
        dropout=0.2,
        compress_dim=64
    ):
        super().__init__()

        if filters_list is None:
            filters_list = [16, 16, 16]

        self.use_compress = compress_dim is not None

        if self.use_compress:
            self.compress = nn.Sequential(
                nn.Linear(input_dim, compress_dim),
                nn.LayerNorm(compress_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            prev_channels = compress_dim
        else:
            self.compress = nn.Identity()
            prev_channels = input_dim

        self.blocks = nn.ModuleList()
        for i, out_channels in enumerate(filters_list):
            self.blocks.append(
                ResBlock(
                    in_channels=prev_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    dilation=2 ** i,
                    dropout=dropout
                )
            )
            prev_channels = out_channels

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(prev_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, seq_len, input_dim)
        x = self.compress(x)         # (B, seq_len, compress_dim) or unchanged
        x = x.permute(0, 2, 1)       # (B, C, seq_len)

        for block in self.blocks:
            x = block(x)

        x = self.pool(x).squeeze(-1)
        x = self.head_dropout(x)
        x = self.fc(x)
        return x


class LightLogModel(nn.Module):
    def __init__(
        self,
        num_classes=2,
        filters_list=None,
        input_dim=300,
        dropout=0.2,
        compress_dim=64
    ):
        super().__init__()
        self.tcn = TCNBackbone(
            input_dim=input_dim,
            num_classes=num_classes,
            filters_list=filters_list,
            dropout=dropout,
            compress_dim=compress_dim
        )

    def forward(self, x):
        return self.tcn(x)