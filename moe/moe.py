import torch
import torch.nn as nn
import torch.nn.functional as F
from .expert import Expert
from .gating import GatingNetwork
import math

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, num_experts, gating_hidden_dim):
        super().__init__()
        self.num_experts = num_experts
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dims, output_dim) 
            for _ in range(num_experts)
        ])

        self.output_layer = nn.Linear(output_dim, 1)
        
        # Create gating network
        self.gate = GatingNetwork(input_dim, gating_hidden_dim, num_experts)
        
        # Initialize tracking metrics
        self.register_buffer('expert_utilization', torch.zeros(num_experts))
        self.total_forward_passes = 0
        self.net = None
        self.projection_matrix = None
    
    def forward(self, x, record=False):

        batch_size = x.shape[0]
        
        # Get expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts])
        # Shape: [num_experts, batch_size, output_dim]


        if True:
            self.net = nn.Sequential(
                nn.Linear(x.shape[-1], 32),
                nn.ReLU(),
                nn.Linear(32, expert_outputs.shape[2]**2)
            ).to('cuda')
            # for param in self.net.parameters():
            #     param.requires_grad = False

    

        
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

        # projection_matrix = self.net(x).view(batch_size, expert_outputs.shape[2], expert_outputs.shape[2])
        # projection_matrix = torch.randn(batch_size, expert_outputs.shape[2], expert_outputs.shape[2]).to('cuda')
        # expert_outputs = batch_project_to_unique_subspaces(
        #     expert_outputs,
        #     projection_matrix
        # )


        if self.projection_matrix is None:
            self.projection_matrix = torch.zeros(expert_outputs.shape[2], expert_outputs.shape[2]).to('cuda')
            torch.nn.init.kaiming_uniform_(self.projection_matrix, a=-math.sqrt(5))



        expert_outputs = project_to_unique_subspaces(
            expert_outputs,
            self.projection_matrix
        )   
        # expert_outputs = gram_schmidt_orthonormalize(expert_outputs)

        cosine_loss = calculate_cosine_loss(expert_outputs)
        lambda_loss = calculate_lambda_max_loss(expert_outputs)


        # expert_outputs = gram_schmidt_orthonormalize(expert_outputs)
        # projected_expert_outputs[:, -1, :] = expert_outputs[:, -1, :]
        # expert_outputs = projected_expert_outputs
        
        # Weighted sum of expert outputs
        combined_output = (expert_outputs * gate_weights).sum(dim=1)
        # Shape: [batch_size, output_dim]

        # Apply final output layer
        final_output = self.output_layer(combined_output)
        
        if record:
            return final_output, cosine_loss, lambda_loss, expert_outputs
        return final_output, cosine_loss, lambda_loss
    
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
    

def batch_project_to_unique_subspaces(
    U: torch.Tensor,
    A: torch.Tensor
) -> torch.Tensor:
    batch, K, dim = U.shape
    # (batch, dim*dim)
    # 2) form a skew-symmetric S(x)
    S = A - A.transpose(-1,-2)             # (batch, dim, dim)
    I = torch.eye(dim, device=U.device).unsqueeze(0)  # (1,dim,dim)

    # 3) Cayley transform per-sample
    #    Q(x) = (I - S)^{-1}(I + S)
    Q = torch.linalg.solve(I - S, I + S)           # (batch, dim, dim)

    # 4) slice into K disjoint blocks and project each expert
    dsub = dim // U.shape[1]
    V = []
    for i in range(U.shape[1]):
        Bi = Q[:, :, i*dsub:(i+1)*dsub]            # (batch, dim, dsub)
        ui = U[:, i].unsqueeze(-1)                 # (batch, dim, 1)
        coords = Bi.transpose(-1,-2) @ ui          # (batch, dsub, 1)
        vi = Bi @ coords                           # (batch, dim, 1)
        V.append(vi.squeeze(-1))                   # (batch, dim)
    V = torch.stack(V, dim=1)                     # (batch, K, dim)
    return V

def project_to_unique_subspaces(
    U: torch.Tensor,
    A: torch.Tensor
) -> torch.Tensor:
    """
    Args:
      U: (batch, K, dim)                — MoE outputs
      A: (dim, dim)                     — unconstrained parameter
    Returns:
      V: (batch, K, dim)                — each expert in its own orthogonal subspace
    """
    batch, K, dim = U.shape
    base, rem = divmod(dim, K)      # e.g. for dim=100, K=6 → base=16, rem=4
    # first `rem` experts get (base+1) dims, the rest get base dims
    sizes = [(base + 1) if i < rem else base for i in range(K)]
    starts = [0] + list(torch.cumsum(torch.tensor(sizes), 0).tolist())

    # build Cayley Q as before
    S = A - A.t()
    I = torch.eye(dim, device=A.device, dtype=A.dtype)
    Q = torch.linalg.solve(I - S, I + S)  # (dim, dim)

    V = torch.zeros_like(U)
    for i in range(K):
        s, e = starts[i], starts[i+1]
        Bi = Q[:, s:e]           # shape (dim, sizes[i])
        ui = U[:, i]             # shape (batch, dim)
        coords = ui @ Bi         # → (batch, sizes[i])
        V[:, i] = coords @ Bi.t()# → (batch, dim)

    # normalise to unit length
    norm = V.norm(dim=2, keepdim=True).clamp_min(1e-6)
    return V / norm




def gram_schmidt_orthonormalize(U: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Differentiable Gram–Schmidt on U of shape (batch, K, dim).
    Avoids in-place ops by cloning and stacking.
    """
    batch, K, dim = U.shape
    orthonorms = []

    for i in range(K):
        # clone the slice so we don't modify a view of U
        v = U[:, i].clone()              # (batch, dim)

        # subtract projections onto all previous orthonormal vectors
        for vj in orthonorms:            # each vj is (batch, dim)
            # ⟨v, vj⟩ / (⟨vj, vj⟩ + eps), shape (batch,1)
            coeff = (v * vj).sum(dim=1, keepdim=True) \
                  / (vj.pow(2).sum(dim=1, keepdim=True) + eps)
            v = v - coeff * vj           # safe: v is a fresh Tensor

        # normalize to unit length
        norm = v.norm(dim=1, keepdim=True).clamp_min(eps)
        v = v / norm

        orthonorms.append(v)

    # stack back into (batch, K, dim)
    return torch.stack(orthonorms, dim=1)



def calculate_cosine_loss(moe_outp):
        '''
        moe output has shape (batch_positions, top_k, dim)
        '''
        # Normalize the tokens along the feature dimension:
        norm_tokens = F.normalize(moe_outp, p=2, dim=-1)  # shape: (batch_positions, top_k, dim)
        
        # Compute cosine similarity matrix for each sample:
        # This produces a (batch_positions, top_k, top_k) tensor where each [i] contains the pairwise similarities.
        cos_sim_matrix = torch.abs(torch.bmm(norm_tokens, norm_tokens.transpose(1, 2)))
        
        # Create a mask to remove self-similarities (the diagonal elements for each sample)
        top_k = moe_outp.size(1)
        diag_mask = torch.eye(top_k, device=moe_outp.device, dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask.expand_as(cos_sim_matrix)
        cos_sim_matrix = cos_sim_matrix.masked_fill(diag_mask, 0)
        
        # Calculate the mean cosine similarity loss per sample.
        # Since each sample has top_k tokens, there are top_k * (top_k - 1) off-diagonals.
        # Sum across the top_k x top_k matrix (which now contains zeros on the diagonal), then average.
        cosine_loss = cos_sim_matrix.sum(dim=(1, 2)) / (top_k * (top_k - 1))
        
        # Finally, take the mean over all batch positions.
        cosine_loss = cosine_loss.mean()
        
        return cosine_loss



def calculate_lambda_max_loss(x):      
        # (batch_positions, top_k, dim) 
        # shapes and dims
        batch_size = x.shape[0]
        dim = x.shape[-1]
        device = x.device

        x = x.permute(1, 2, 0)  # (batch_positions, dim, top_k)


        x = F.normalize(x, p=2, dim=1)

        eps = 1e-3

        Q, R = torch.linalg.qr(x, mode="reduced")

            
        r_diag = R.abs().diagonal(dim1=-2, dim2=-1)           # (E, min(d,B))
        k      = (r_diag > eps).sum(dim=1)           
        # for i, ki in enumerate(k):
        #     print(f"expert_{i}_empirical_rank", ki.item())         
        cols   = torch.arange(Q.size(-1), device=Q.device)    # (d,)
        mask   = cols[None, None, :] < k[:, None, None]       # (E, 1, d)
        Qm     = Q * mask                                     
        projs  = Qm @ Qm.transpose(-2, -1) 
        avg_proj    = projs.mean(dim=0) 

        eigvals = torch.linalg.eigvalsh(avg_proj)
        lambda_max = eigvals[-1]
    
        return lambda_max.mean()
