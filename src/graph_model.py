import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class GCNConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads, concat=True):
        super().__init__()

        self.conv = TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat
        )

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


class GraphTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        embed_dim=128,
        heads=4,
        dropout_rate=0.2
    ):
        super().__init__()

        self.conv1 = GCNConvLayer(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            concat=True
        )

        self.conv2 = GCNConvLayer(
            in_channels=hidden_channels * heads,
            out_channels=embed_dim,
            heads=1,
            concat=False
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)

        # molecule-level graph embedding
        x = x.mean(dim=0, keepdim=True)

        return x