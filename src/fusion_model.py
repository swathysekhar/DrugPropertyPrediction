import torch
import torch.nn as nn
class CrossAttentionFusion(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        dropout=0.3
    ):
        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 1)
        )

    def forward(
        self,
        graph_emb,
        smiles_emb,
        fg_emb
    ):

        # =================================================
        # Query = Graph
        # =================================================

        query = graph_emb.unsqueeze(1)      # [1,1,128]

        # =================================================
        # Key/Value = SMILES + FG
        # =================================================

        key_value = torch.stack(
            [
                smiles_emb.squeeze(0),
                fg_emb.squeeze(0)
            ],
            dim=0
        ).unsqueeze(0)                      # [1,2,128]

        # =================================================
        # Cross Attention
        # =================================================

        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=key_value,
            value=key_value
        )

        # =================================================
        # Residual Connection
        # =================================================

        fused = attn_output.squeeze(1)      # [1,128]

        fused = self.norm(fused + graph_emb)

        # =================================================
        # Classification
        # =================================================

        logits = self.classifier(fused)     # [1,1]

        return logits, attn_weights


class FusionModel(nn.Module):

    def __init__(
        self,
        graph_model,
        sequence_model,
        fg_encoder,
        embed_dim=128
    ):
        super().__init__()

        self.graph_model = graph_model
        self.sequence_model = sequence_model
        self.fg_encoder = fg_encoder

        self.cross_fusion = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=0.3
        )

    def forward(
        self,
        graph_data,
        sequence_inputs,
        fg
    ):
        device = next(self.parameters()).device

        graph_data = graph_data.to(device)
        sequence_inputs = sequence_inputs.to(device)
        fg = fg.to(device)
        graph_emb = self.graph_model(graph_data)

        smiles_emb = self.sequence_model(sequence_inputs)

        fg_emb = self.fg_encoder(fg)

        logits, attn_weights = self.cross_fusion(
            graph_emb,
            smiles_emb,
            fg_emb
        )

        return logits, attn_weights