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
    
    def forward(self, x):

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

        #cosine_loss = calculate_cosine_loss(expert_outputs)

        expert_outputs = gram_schmidt_orthonormalize(expert_outputs)
        # projected_expert_outputs[:, -1, :] = expert_outputs[:, -1, :]
        # expert_outputs = projected_expert_outputs
        cosine_loss = 0 
        
        # Weighted sum of expert outputs
        combined_output = (expert_outputs * gate_weights).sum(dim=1)
        # Shape: [batch_size, output_dim]

        # Apply final output layer
        final_output = self.output_layer(combined_output)
        
        return final_output, cosine_loss
    
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


    A = A.to('cpu')
    Q, R = torch.linalg.qr(A, mode='complete')  # (dim, dim), (dim, dim)
    Q = Q.to('cuda')

    V = torch.zeros_like(U)
    for i in range(K):
        s, e = starts[i], starts[i+1]
        Bi = Q[:, s:e]           # shape (dim, sizes[i])
        ui = U[:, i]             # shape (batch, dim)
        coords = ui @ Bi         # → (batch, sizes[i])
        V[:, i] = coords @ Bi.t()# → (batch, dim)
    return V



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





def torch_qr(a, mode='complete', out=None, gram='classical'):
    """
    Due to a bug in MAGMA, qr on cuda is super slow for small matrices. 
    Therefore, this step must be performed on the cpu.
   

    This function aims to provide a temporary relief for using 
    torch.linalg.qr on GPU by implementing a Gram-Schmidt process. 
    
    Note: This implementation does not support backward propagation, and 
          only supports the 'complete' mode.
    
    See the following regarding this Bug:
        https://github.com/pytorch/pytorch/issues/22573
        https://github.com/cornellius-gp/gpytorch/pull/1224
        
    The input arguments, other than 'gram', follow the PyTorch standard. 
    See the following for their definition:
        https://pytorch.org/docs/stable/generated/torch.linalg.qr.html
        
    Parameters
    ----------
    a: (torch.tensor) the input tensor. Must have a shape of 
        (*mb_dims, dim, dim), where mb_dims shows the batch 
        dimensions.

    mode: (str) Either 'complete' or 'reduced'. This current 
        implementation only supports the former.
        
    out: (None or torch.tensor) The output tensor for the Q matrix. 
        If provided, must have the same shape as a.
        
    gram: (str) The Gram-Schmidt process variant. 
    
        * The classical variant makes O(dim) calls to CUDA 
          and can be more efficient. 
          
        * The modified variant can be slightly more accurate, 
          but makes CUDA O(dim^2) calls and thus is less efficient.
          
          See Section 14.2 of "Numerical Linear Algebra with Applications" 
          by William Ford on the numerical stability of Gram-Schmidt and 
          its modified variant:
          
          https://www.sciencedirect.com/science/article/abs/pii/B9780123944351000144
          
        * The cpu variant uses Pytorch's routine on CPU.
          
        This has to be one of ('classical', 'modified', 'cpu').
        
    Output
    ------
    q: (torch.tensor) The output orthonormal matrix. 
        This should have a shape of (*mb_dims, dim, dim).
    
    r: (torch.tensor) The output upper triangle matrix. 
        This should have a shape of (*mb_dims, dim, dim).
    """
    # First Solution: Performing the QR decomposition on CPU
    # Issues: 
    #    1. Pytorch may still only utilize one thread 
    #       practically even though torch.get_num_threads() 
    #       may be large.
    #    2. Reliance on CPU resources.
    if gram == 'cpu':
        q, r = torch.linalg.qr(a.detach().cpu(), mode=mode, out=out)
        return q.to(device=a.device), r.to(device=a.device)
    
    ###############################################################
    ################## Initializing & Identifying #################
    ###############################################################
    assert mode == 'complete', 'reduced is not implemented yet'
    # The bactch dimensions
    mb_dims = a.shape[:-2]
    # The input device
    tch_device = a.device
    
    # The Data Type for performing the mathematical caculations
    # Note: Gram-schmidt is numerically unstable. For this reason, even 
    # when the input may be float32, we will do everything in float64.
    tch_dtype = torch.float64
    
    # The QR process dimension
    dim = a.shape[-1]
    assert a.shape == (*mb_dims, dim, dim)

    if out is None:
        q = torch.empty(*mb_dims, dim, dim, device=tch_device, dtype=tch_dtype)
    else:
        q = out
    assert q.shape == (*mb_dims, dim, dim)
    
    # Casting the a input to tch_dtype and using it from now on
    a_f64 = a.to(dtype=tch_dtype)
    a_f64.requires_grad_(a.requires_grad)

    
    ###############################################################
    ################### Performing Gram-Schmidt ###################
    ###############################################################
    if gram == 'classical':
        # Performing the classical Gram-Schmidt Process.
        
        # Creating a copy of a to avoid messing up the original input
        acp = a_f64.clone()
        assert acp.shape == (*mb_dims, dim, dim)
        
        for k in range(dim):
            qk_unnorm = acp[..., :, k:k+1]
            assert qk_unnorm.shape == (*mb_dims, dim, 1)

            qk = qk_unnorm / qk_unnorm.norm(dim=-2, keepdim=True)
            assert qk.shape == (*mb_dims, dim, 1)

            a_qkcomps = qk.reshape(*mb_dims, 1, dim).matmul(acp)
            assert a_qkcomps.shape == (*mb_dims, 1, dim)

            # Removing the qk components from a
            acp -= qk.matmul(a_qkcomps)
            assert acp.shape == (*mb_dims, dim, dim)

            q[..., :, k] = qk.reshape(*mb_dims, dim)
    elif gram == 'modified':
        # Performing the modified Gram-Schmidt Process.
        for i in range(dim):
            q[..., i] = a_f64[..., i]
            for j in range(i):
                err_ij = torch.einsum('...i,...i->...', q[..., j], q[..., i])
                assert err_ij.shape == (*mb_dims,)
                q[..., i] -=  err_ij.reshape(*mb_dims, 1) * q[..., j]
            q[..., i] /= q[..., i].norm(dim=-1, keepdim=True)
    else:
        raise ValueError(f'Unknown gram={gram}')

    r = q.transpose(-1, -2).matmul(a_f64)
    assert r.shape == (*mb_dims, dim, dim)

    ###############################################################
    ######################## Final Cleanup ########################
    ###############################################################
    # Making sure the lower triangle of r is absolutely zero!
    col = torch.arange(dim, device=tch_device, dtype=tch_dtype).reshape(1, dim)
    assert col.shape == (1, dim)

    row = col.reshape(dim, 1)
    assert row.shape == (dim, 1)
    
    mb_ones = [1] * len(mb_dims)
    r *= (row <= col).reshape(*mb_ones, dim, dim)
    
    # Casting the q and r outputs to the a input dtype for compatibility
    q_out, r_out = q.to(dtype=a.dtype), r.to(dtype=a.dtype)
    
    return q_out, r_out


