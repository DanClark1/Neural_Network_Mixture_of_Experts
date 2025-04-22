import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Hidden layers with skip connections
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in hidden_dims
        ])
        
    def forward(self, x):
        # Initial layer
        h = F.relu(self.layers[0](x))
        
        # Hidden layers with skip connections and layer norm
        for i in range(1, len(self.layers) - 1):
            h_prev = h
            h = self.layers[i](h)
            h = self.layer_norms[i-1](h)
            h = F.relu(h)
            if h.shape == h_prev.shape:  # Add skip connection if shapes match
                h = h + h_prev

        print(f"Expert forward pass: {h.shape}")
        
        # Output layer
        return self.layers[-1](h)