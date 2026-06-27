#code3.py
#1. The Manual MoRE Framework Wrapper (Tensor Realignment)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEBlock(nn.Module):
    def __init__(self, num_experts=7, hidden_dim=256):
        super().__init__()
        # 7 experts matching your operation families
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        # x shape: [NumActiveTokens, HiddenDim]
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        expert_idx = torch.argmax(router_probs, dim=-1)
        
        out = torch.zeros_like(x)
        
        # Route tokens dynamically to their specific operation expert
        for i, expert in enumerate(self.experts):
            mask = (expert_idx == i)
            if mask.any():
                out[mask] = expert(x[mask])
                
        # Simple entropy balancing loss to prevent dead experts
        avg_probs = router_probs.mean(dim=0)
        balance_loss = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
        
        return out, balance_loss

class MoREWrapper(nn.Module):
    def __init__(self, hidden_dim=256, max_depth=7, num_experts=7):
        super().__init__()
        self.max_depth = max_depth
        self.moe_block = MoEBlock(num_experts, hidden_dim)
        # Halting router predicts the probability of stopping
        self.halting_router = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: [BatchSize, SeqLen, HiddenDim]
        B, S, D = x.shape
        flat_x = x.view(B * S, D)
        
        current_state = flat_x.clone()
        final_output = torch.zeros_like(flat_x)
        
        # Binary mask tracker: True means the token keeps recursing
        active_tokens = torch.ones(B * S, dtype=torch.bool, device=x.device)
        
        total_balance_loss = 0.0
        total_halt_loss = 0.0
        
        for depth in range(1, self.max_depth + 1):
            if not active_tokens.any():
                break
                
            # 1. Process only active tokens through the MoE block
            active_inputs = current_state[active_tokens]
            moe_out, b_loss = self.moe_block(active_inputs)
            
            # 2. Residual connection update
            current_state[active_tokens] = active_inputs + moe_out
            total_balance_loss += b_loss
            
            # 3. Predict halting decision
            halt_logits = self.halting_router(current_state[active_tokens]).squeeze(-1)
            halt_probs = torch.sigmoid(halt_logits)
            
            # Token exits if probability > 0.5
            stop_decision = halt_probs > 0.5
            
            # Identify tokens that are stopping exactly at this depth step
            stopping_now = active_tokens.clone()
            stopping_now[active_tokens] = stop_decision
            
            # Force remaining active tokens to exit if we hit max depth limit
            if depth == self.max_depth:
                stopping_now[active_tokens] = True
                
            # 4. Lock in state for exiting tokens
            final_output[stopping_now] = current_state[stopping_now]
            
            # 5. Evict stopped tokens from the next loop iteration
            active_tokens[stopping_now] = False
            
            # Small penalty loss to naturally encourage early exiting
            total_halt_loss += halt_probs.mean() * 0.01

        return final_output.view(B, S, D), total_balance_loss, total_halt_loss

# Quickly verify shapes locally before launching
if __name__ == "__main__":
    model = MoREWrapper()
    dummy_input = torch.randn(16, 128, 256) # Batch=16, Seq=128, Dim=256
    out, b_loss, h_loss = model(dummy_input)
    print("Output shape aligned:", out.shape) # Expect [16, 128, 256]