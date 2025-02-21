import torch
import torch.nn as nn
import torch.nn.functional as F

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
        # Smaller network than experts
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
    
    def forward(self, x):
        # Compute gating weights with temperature scaling
        logits = self.net(x) / self.temperature
        return F.softmax(logits, dim=-1)