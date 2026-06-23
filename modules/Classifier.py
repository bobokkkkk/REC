from torch import nn
class DeepMLPClassifier(nn.Module):
    def __init__(self, in_channels=768, dropout=0.9):
        super().__init__()

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.feature_selector = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.8),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.4),

            nn.Linear(128, 2)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        attention = self.feature_selector(x)
        x = x * attention
        res = self.mlp(x)
        return res