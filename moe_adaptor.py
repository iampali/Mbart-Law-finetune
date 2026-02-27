import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, n_embed=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n_embed, out_features=4*n_embed),
            nn.ReLU(),
            nn.Linear(in_features=4*n_embed, out_features=n_embed),
            nn.Dropout(p=0.2, inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class NoisyTopk(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopk, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(n_embed, num_experts)
        self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        logits = self.linear(mh_output)
        noise = self.noise_linear(mh_output)

        # FIX 1: Force softplus back to bfloat16 immediately!
        # This stops the float32 promotion from infecting noisy_logits
        safe_noise = F.softplus(noise).to(logits.dtype)
        
        noisy_logits = logits + torch.randn_like(logits) * safe_noise

        ## choosing top_k from noisy_logits
        top_k_noisy_logits, top_k_noisy_indices = noisy_logits.topk(self.top_k, dim=-1)
        
        # FIX 2: Create the matrix explicitly matching noisy_logits
        inf_matrix = torch.full_like(noisy_logits, float('-inf'))

        # This will now work perfectly because both are guaranteed bfloat16
        sparse_noisy_logits = inf_matrix.scatter(dim=-1, index=top_k_noisy_indices, src=top_k_noisy_logits)

        # FIX 3: Softmax also gets promoted to float32 by autocast. 
        # We must cast the final routing probabilities back to bfloat16 
        # so they don't crash your expert matrix multiplications later!
        routing_probs = F.softmax(sparse_noisy_logits, dim=-1).to(logits.dtype)

        return routing_probs, top_k_noisy_indices
    
class SparseMOE(nn.Module):
    def __init__(self, n_embed, num_experts, top_K):
        super(SparseMOE, self).__init__()
        self.top_k = top_K
        self.router = NoisyTopk(n_embed=n_embed, num_experts=num_experts, top_k=top_K)
        self.experts = nn.ModuleList([Expert(n_embed=n_embed) for _ in range(num_experts)])

    def forward(self, x):
        gating_output, indices = self.router(x) # 2, 4, 3 (bs, sl, num_experts)
        final_output = torch.zeros_like(x) # 2,4,8 (bs, sl, embed_size)

        ## Reshape inputs for batch processing
        flat_x = x.reshape(-1, x.size(-1)) # 8,8 (bs*sl, embed_size)
        flat_gating_output = gating_output.reshape(-1, gating_output.size(-1)) # 8, 3 (bs*sl, num_experts)

        # Preprocess each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1) # 2, 4 (bs, sl)
            flat_mask = expert_mask.reshape(-1) # 8 (bs * sl)
  
            if flat_mask.any():
                expert_input = flat_x[flat_mask] # depends on how many True's are there, let's say 6 tokens will moving to the expert i. (6, 8) (selected num_tokens from bs * sl, embed_size)
                expert_output = expert(expert_input) # It's MLP layer will give the output same as input size. so (6, 8)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(dim=1) # (6,1), (selected num_tokens from bs * sl, unsqueeze_dim)
                weighted_output = expert_output * gating_scores # (6, 8) (selected num_tokens from bs * sl, embed_size)

                # update final output additvely by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(dim=1) #(batch_size,sl,embed_size)
        
        return final_output


