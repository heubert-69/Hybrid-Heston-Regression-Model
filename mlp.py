import torch.nn as nn

class MLPModule(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=2, dropout=0.2, output_dim=5):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))  
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)