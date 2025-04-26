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
        total_loss = task_loss + self.load_balance_coef * load_balance_loss + lambda_loss * 0.1
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
            for x, y in val_loader:
                x = x.to('cuda')
                y = y.to('cuda')
                outputs, cosine_loss, lambda_loss, *_ = self.model(x, record=record)
                cosine_losses.append(cosine_loss)
                lambda_losses.append(lambda_loss)
                if record:
                    _, Z = _
                    Z_full.append(Z)
                loss = self.task_loss_fn(outputs, y)
                total_loss += loss.item()
                num_batches += 1

        if record:
            #Z_full = torch.cat(Z_full, dim=0).cpu().numpy()
            Z_full  = Z.cpu().numpy()
            out_root = '/app/save_dir/'
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join(out_root, ts)
            os.makedirs(out_dir, exist_ok=True)

            # 2) center and SVD
            Zc = Z_full - Z_full.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(Zc, full_matrices=False)

            # 3) scree plot
            plt.figure()
            plt.plot(S, marker='o')
            plt.yscale('log')
            plt.xlabel("Component index")
            plt.ylabel("Singular value (log scale)")
            plt.title("Scree Plot of Singular Values")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "scree_plot.png"))
            plt.close()

            # 4) cumulative explained variance
            explained = S**2
            cumul = np.cumsum(explained) / np.sum(explained)
            plt.figure()
            plt.plot(cumul, marker='o')
            plt.xlabel("Number of components")
            plt.ylabel("Cumulative explained variance")
            plt.title("Cumulative Explained Variance")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "cumulative_explained_variance.png"))
            plt.close()


        avg_cosine_loss = sum(cosine_losses) / len(cosine_losses)
        avg_lambda_loss = sum(lambda_losses) / len(lambda_losses)
        print('Average cosine loss:', avg_cosine_loss)
        print('Average lambda loss:', avg_lambda_loss)
        
        self.model.train()
        return total_loss / num_batches