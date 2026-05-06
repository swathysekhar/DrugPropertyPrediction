import torch
import torch.nn as nn
class FunctionalGroupEmbedding(nn.Module):
    def __init__(
        self,
        fg_input_dim,
        embed_dim=128,
        dropout_rate=0.2
    ):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(fg_input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(128, embed_dim)
        )

    def forward(self, fg):

        if fg.dim() == 1:
            fg = fg.unsqueeze(0)

        fg = fg.float()

        return self.fc(fg)