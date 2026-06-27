import wandb
import torch
import torch.optim as optim

# Initialize W&B run
wandb.init(
    project="micro-MoRE-poc",
    config={
        "d_model": 128,
        "num_experts": 4,
        "max_depth": 3,
        "lr": 1e-3,
        "batch_size": 32,
        "seq_len": 16
    }
)

# Setup model, optimizer, and dummy data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MoRELayer(d_model=128, num_experts=4, max_depth=3).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Simple loop to demonstrate metric tracking
for epoch in range(100):
    # Dummy sequence representing incoming feature vectors
    dummy_input = torch.randn(32, 16, 128, device=device)
    target = dummy_input * 1.5 # Arbitrary target task
    
    optimizer.zero_grad()
    
    # Forward pass outputs the tensor, routing loss, and load entropy
    output, aux_loss, entropy = model(dummy_input)
    
    # Primary task loss (MSE for demonstration)
    task_loss = F.mse_loss(output, target)
    
    # Combined loss
    total_loss = task_loss + 0.1 * aux_loss
    
    total_loss.backward()
    optimizer.step()
    
    # Calculate Max Pairwise Cosine Similarity between Expert weights manually
    with torch.no_grad():
        w_exp0 = model.experts[0].w1.weight.flatten()
        w_exp1 = model.experts[1].w1.weight.flatten()
        max_cos_sim = F.cosine_similarity(w_exp0, w_exp1, dim=0).item()

    # Log everything to W&B
    wandb.log({
        "train/total_loss": total_loss.item(),
        "train/task_loss": task_loss.item(),
        "train/aux_routing_loss": aux_loss.item(),
        "train/expert_load_entropy": entropy.item(),
        "diag/max_pairwise_cosine_sim": max_cos_sim
    })

wandb.finish()