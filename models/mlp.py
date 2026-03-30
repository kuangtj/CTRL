import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super(MLP, self).__init__()
        
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))  # Output layer (no activation here if using CrossEntropyLoss)
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)