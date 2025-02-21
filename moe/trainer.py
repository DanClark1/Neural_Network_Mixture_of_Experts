import torch
import torch.nn as nn

class MoETrainer:
    def __init__(self, model, optimizer, task_loss_fn, load_balance_coef=0.1):
        self.model = model
        self.optimizer = optimizer
        self.task_loss_fn = task_loss_fn
        self.load_balance_coef = load_balance_coef
        
    def train_step(self, x, y):
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
        
        with torch.no_grad():
            for x, y in val_loader:
                outputs = self.model(x)
                loss = self.task_loss_fn(outputs, y)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches