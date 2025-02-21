import torch
import torch.nn as nn
import torch.nn.functional as F
from .expert import Expert
from .gating import GatingNetwork

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_experts, gating_hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dims, output_dim) 
            for _ in range(num_experts)
        ])
        
        # Create gating network
        self.gate = GatingNetwork(input_dim, gating_hidden_dim, num_experts)
        
        # Initialize tracking metrics
        self.register_buffer('expert_utilization', torch.zeros(num_experts))
        self.total_forward_passes = 0
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Get expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        # Shape: [num_experts, batch_size, output_dim]
        
        # Get gating weights
        gate_weights = self.gate(x)
        # Shape: [batch_size, num_experts]
        
        # Update utilization metrics
        with torch.no_grad():
            self.expert_utilization += gate_weights.sum(0)
            self.total_forward_passes += batch_size
        
        # Combine expert outputs
        # Reshape gate_weights for broadcasting
        gate_weights = gate_weights.unsqueeze(-1)
        # Shape: [batch_size, num_experts, 1]
        
        # Permute expert outputs for batch-first
        expert_outputs = expert_outputs.permute(1, 0, 2)
        # Shape: [batch_size, num_experts, output_dim]
        
        # Weighted sum of expert outputs
        combined_output = (expert_outputs * gate_weights).sum(dim=1)
        # Shape: [batch_size, output_dim]
        
        return combined_output
    
    def get_expert_utilization_rates(self):
        if self.total_forward_passes == 0:
            return torch.zeros_like(self.expert_utilization)
        return self.expert_utilization / self.total_forward_passes
    
    def compute_load_balancing_loss(self, gate_weights):
        # Compute load balancing loss to prevent expert collapse
        # Calculate mean utilization per expert in this batch
        expert_utilization = gate_weights.mean(0)
        # Ideal uniform utilization
        uniform_utilization = torch.ones_like(expert_utilization) / self.num_experts
        # Calculate KL divergence from uniform
        load_balancing_loss = F.kl_div(
            expert_utilization.log(),
            uniform_utilization,
            reduction='batchmean'
        )
        return load_balancing_loss