import torch
import torch.nn as nn
import torch.nn.functional as F

class RecursiveExpert(nn.Module):
    """An expert that processes tokens recursively using shared weights."""
    def __init__(self, d_model, d_ff, max_depth=3):
        super().__init__()
        self.max_depth = max_depth
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        
    def forward(self, x):
        # The token passes through the exact same layers recursively
        for _ in range(self.max_depth):
            residual = x
            x = self.w2(self.act(self.w1(x)))
            x = x + residual  # Layer-internal residual connection
        return x

class MoRELayer(nn.Module):
    """Mixture of Recursive Experts Layer with Top-2 Routing."""
    def __init__(self, d_model, num_experts=4, max_depth=3):
        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            RecursiveExpert(d_model, d_ff=d_model*4, max_depth=max_depth) 
            for _ in range(num_experts)
        ])
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        b, s, d = x.shape
        x_flat = x.view(-1, d)  # [total_tokens, d_model]
        
        # 1. Compute routing logits
        router_logits = self.router(x_flat)  # [total_tokens, num_experts]
        
        # 2. Top-2 Routing
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, k=2, dim=-1)
        
        # Normalize top-2 weights so they sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # 3. Compute Auxiliary Balancing Loss (to prevent representation collapse)
        # Ideal distribution is uniform across all experts
        out_tokens = torch.zeros_like(x_flat)
        expert_counters = torch.zeros(self.num_experts, device=x.device)
        
        # 4. Dispatch tokens to selected experts
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert (either as top-1 or top-2)
            mask = (topk_indices == i)
            if not mask.any():
                continue
                
            token_indices, topk_positions = torch.where(mask)
            selected_tokens = x_flat[token_indices]
            
            # Run the selected tokens through the recursive expert
            expert_outputs = expert(selected_tokens)
            
            # Apply weights and accumulate back
            weights = topk_weights[token_indices, topk_positions].unsqueeze(-1)
            out_tokens.index_add_(0, token_indices, expert_outputs * weights)
            expert_counters[i] += token_indices.numel()
            
        # Calculate routing metrics to return for W&B
        expert_load = expert_counters / expert_counters.sum()
        entropy = -torch.sum(expert_load * torch.log(expert_load + 1e-6))
        
        # Basic auxiliary loss component (variance of load)
        aux_loss = torch.var(expert_load) 
        
        return out_tokens.view(b, s, d), aux_loss, entropy