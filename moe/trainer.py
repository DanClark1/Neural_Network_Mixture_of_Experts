import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np

class MoETrainer:
    def __init__(self, model, optimizer, task_loss_fn, load_balance_coef=0.1):
        self.model = model
        self.optimizer = optimizer
        self.task_loss_fn = task_loss_fn
        self.load_balance_coef = load_balance_coef
        
    def train_step(self, x, y):
        self.optimizer.zero_grad()
        x = x.to('cuda')
        y = y.to('cuda')
        # Forward pass
        outputs, cosine_loss, lambda_loss = self.model(x)
        gate_weights = self.model.gate(x)
        
        # Compute losses
        task_loss = self.task_loss_fn(outputs, y)
        load_balance_loss = self.model.compute_load_balancing_loss(gate_weights)
        
        # Combined loss
        total_loss = task_loss + self.load_balance_coef * load_balance_loss# + cosine_loss * 0.1 #+ lambda_loss * 0.1
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
    
    def evaluate(self, val_loader, record=False):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        cosine_losses = []
        lambda_losses = []
        with torch.no_grad():
            Z_full = []
            list_of_e_outputs = [[] for _ in range(self.model.num_experts)]
            for x, y in val_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                outputs, cosine_loss, lambda_loss, *expert_outputs = self.model(x, record=record)
                cosine_losses.append(cosine_loss)
                lambda_losses.append(lambda_loss)
                
                if record:
                    e_outputs = expert_outputs[0].permute(1, 2, 0) # (dim, experts, batch) -> (experts, batch, dim)
                    for i in range(e_outputs.shape[0]):
                        # move the expert to the origin
                        X = e_outputs[i]
                        X_norm = X / torch.linalg.norm(X, dim=-1, keepdim=True)
                        mean = X_norm.mean(dim=0, keepdim=True)
                        X_centered = X_norm - mean
                        cov = (X_centered.T @ X_centered) / (X_norm.shape[0] - 1)
                        sign, logabsdet = torch.linalg.slogdet(cov)
                        gen_var = sign * torch.exp(logabsdet)
                        list_of_e_outputs[i].append(gen_var.cpu())




                loss = self.task_loss_fn(outputs, y)
                total_loss += loss.item()
                num_batches += 1

            if record:
                for e in range(len(list_of_e_outputs)):
                    list_of_e_outputs[e] = torch.sum(torch.stack(list_of_e_outputs[e])) / num_batches
                    print(f"Expert {e} output variance: {list_of_e_outputs[e]}")            


        avg_cosine_loss = sum(cosine_losses) / len(cosine_losses)
        avg_lambda_loss = sum(lambda_losses) / len(lambda_losses)
        print('Average cosine loss:', avg_cosine_loss)
        print('Average lambda loss:', avg_lambda_loss)
        
        self.model.train()
        return total_loss / num_batches