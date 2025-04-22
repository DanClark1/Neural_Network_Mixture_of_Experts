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
        
        # Output layer
        return self.layers[-1](h)

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

# Training utilities
class MoETrainer:
    def __init__(self, model, optimizer, task_loss_fn, load_balance_coef=0.1):
        self.model = model
        self.optimizer = optimizer
        self.task_loss_fn = task_loss_fn
        self.load_balance_coef = load_balance_coef
        
    def train_step(self, x, y):
        # ensure x and y are on the same device as the model
        device = next(self.model.parameters()).device
        x, y = x.to(device), y.to(device)

        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(x)
        gate_weights = self.model.gate(x)
        
        # Compute losses
        task_loss = self.task_loss_fn(outputs, y)
        load_balance_loss = self.model.compute_load_balancing_loss(gate_weights)
        
        # Combined loss
        total_loss = task_loss + self.load_balance_coef * load_balance_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return {
            'task_loss': task_loss.item(),
            'load_balance_loss': load_balance_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for x, y in val_loader:
                # move batch to GPU if needed
                x, y = x.to(device), y.to(device)
                outputs = self.model(x)
                loss = self.task_loss_fn(outputs, y)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches