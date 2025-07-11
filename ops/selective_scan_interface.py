"""Mock implementation of selective_scan_fn for testing purposes.

This file provides a simple PyTorch implementation of the selective_scan function
that is normally provided by the BlackMamba CUDA extension.
"""

import torch
import torch.nn.functional as F

def selective_scan_fn(
    u, delta, A, B, C, D=None,
    delta_bias=None, delta_softplus=False,
    return_last_state=False
):
    """Mock implementation of selective_scan for testing.
    
    Args:
        u: Input tensor of shape (B, D, L)
        delta: Delta tensor of shape (B, D, L)
        A: A tensor of shape (D, N)
        B: B tensor of shape (B, G, N, L) where G is num_heads
        C: C tensor of shape (B, G, N, L) where G is num_heads
        D: D tensor of shape (D,), optional
        delta_bias: Bias for delta, optional
        delta_softplus: Whether to apply softplus to delta
        return_last_state: Whether to return the last state
        
    Returns:
        y: Output tensor of shape (B, D, L)
        last_state: Last state tensor of shape (B, D, N) if return_last_state=True
    """
    batch_size, dim, seq_len = u.shape
    state_size = A.shape[1]
    num_heads = B.shape[1]
    head_dim = dim // num_heads
    
    # Apply softplus to delta if requested
    if delta_softplus:
        delta = F.softplus(delta)
    
    # Add bias to delta if provided
    if delta_bias is not None:
        delta = delta + delta_bias
    
    # Initialize state - one per head
    x = torch.zeros(batch_size, num_heads, state_size, device=u.device, dtype=u.dtype)
    
    # Initialize output
    y = torch.zeros_like(u)
    
    # Reshape u and delta to match the head structure
    # u: (B, D, L) -> (B, G, H, L) where D = G*H
    u_reshaped = u.view(batch_size, num_heads, head_dim, seq_len)
    # delta: (B, D, L) -> (B, G, H, L)
    delta_reshaped = delta.view(batch_size, num_heads, head_dim, seq_len)
    
    # Reshape A to match the head structure
    # A: (D, N) -> (G, H, N)
    A_reshaped = A.view(num_heads, head_dim, state_size)
    
    # Process each timestep
    for t in range(seq_len):
        # Get current timestep tensors
        u_t = u_reshaped[:, :, :, t]  # (B, G, H)
        delta_t = delta_reshaped[:, :, :, t]  # (B, G, H)
        B_t = B[:, :, :, t]  # (B, G, N)
        C_t = C[:, :, :, t]  # (B, G, N)
        
        # Process each head
        for h in range(num_heads):
            # Get head-specific tensors
            u_h = u_t[:, h]  # (B, H)
            delta_h = delta_t[:, h]  # (B, H)
            A_h = A_reshaped[h]  # (H, N)
            B_h = B_t[:, h]  # (B, N)
            C_h = C_t[:, h]  # (B, N)
            x_h = x[:, h]  # (B, N)
            
            # Compute A*delta for each batch and state
            # delta_h: (B, H), A_h: (H, N)
            A_delta = torch.matmul(delta_h, A_h)  # (B, N)
            
            # Update state: x = exp(A*delta)*x + B*u
            # Compute B*u: B_h: (B, N), u_h: (B, H)
            # Need to average u_h across head dimension for now
            u_h_avg = u_h.mean(dim=1, keepdim=True)  # (B, 1)
            Bu = B_h * u_h_avg  # (B, N)
            
            # Update state
            x[:, h] = torch.exp(A_delta) * x_h + Bu
            
            # Compute output contribution: y = C*x
            y_h = torch.sum(C_h * x[:, h], dim=1)  # (B,)
            
            # Add to output at the right position
            start_idx = h * head_dim
            end_idx = (h + 1) * head_dim
            y[:, start_idx:end_idx, t] = y_h.unsqueeze(1).expand(-1, head_dim)
    
    # Add direct connection if D is provided
    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1)  # D is (D,), expand to (1, D, 1)
    
    # Return last state if requested
    if return_last_state:
        return y, x
    else:
        return y
