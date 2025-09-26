import torch.nn as nn

class AxiomModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AxiomModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)
