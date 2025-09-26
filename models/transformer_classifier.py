import torch
import torch.nn as nn

class TransformerTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, hidden_dim, num_layers, num_classes, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x).permute(1, 0, 2)  # (seq_len, batch, embed_dim)
        out = self.transformer(emb)
        out = out.mean(dim=0)  # mean pooling
        return self.fc(out)
