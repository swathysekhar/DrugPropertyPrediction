import torch
import torch.nn as nn


class SMILESTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        dim_feedforward=512,
        max_length=100,
        dropout=0.3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0
        )

        self.pos_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

    def forward(self, sequence):
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)

        sequence = sequence.long()

        batch_size, seq_len = sequence.shape

        positions = torch.arange(seq_len, device=sequence.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        x = self.embedding(sequence) + self.pos_embedding(positions)

        padding_mask = sequence.eq(0)

        x = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )

        mask = (~padding_mask).float().unsqueeze(-1)

        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return x