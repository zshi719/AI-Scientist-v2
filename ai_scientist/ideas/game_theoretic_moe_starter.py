"""
Game-Theoretic Expert Selection for Mixture-of-Experts
This implements the core algorithmic components for testing the hypothesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import bisect


class TsallisRouter(nn.Module):
    """Game-theoretic router with Tsallis entropy regularization"""
    
    def __init__(self, input_dim, num_experts, q=2.0, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.q = q
        self.temperature = temperature
        
        # Utility predictor network
        self.utility_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_experts)
        )
        
    def compute_threshold(self, utilities):
        """Find threshold τ such that routing probabilities sum to 1"""
        def constraint(tau):
            if abs(self.q - 1.0) < 1e-6:
                return 1.0  # For q→1, use softmax (no threshold needed)
            clipped = torch.clamp((utilities / self.temperature - tau), min=0)
            probs = clipped ** (1 / (self.q - 1))
            return probs.sum().item() - 1.0
        
        # Binary search for threshold
        u_max = utilities.max().item() / self.temperature
        u_min = utilities.min().item() / self.temperature - 10
        
        try:
            tau = bisect(constraint, u_min, u_max, xtol=1e-6)
        except:
            tau = utilities.mean().item() / self.temperature
            
        return tau
    
    def forward(self, x):
        # Compute utilities
        utilities = self.utility_net(x)
        
        # Game-theoretic routing probabilities
        if abs(self.q - 1.0) < 1e-6:
            # Shannon entropy → softmax
            return F.softmax(utilities / self.temperature, dim=-1)
        else:
            # Tsallis entropy → sparse routing
            batch_size = utilities.shape[0]
            probs = torch.zeros_like(utilities)
            
            for b in range(batch_size):
                tau = self.compute_threshold(utilities[b])
                clipped = torch.clamp(
                    (utilities[b] / self.temperature - tau), min=0
                )
                probs[b] = clipped ** (1 / (self.q - 1))
                probs[b] = probs[b] / (probs[b].sum() + 1e-8)
                
            return probs


class GameTheoreticMoE(nn.Module):
    """Mixture of Experts with game-theoretic routing"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts=8, q=2.0):
        super().__init__()
        
        self.router = TsallisRouter(input_dim, num_experts, q=q)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, output_dim)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x):
        # Get routing probabilities (Nash equilibrium strategy)
        routing_probs = self.router(x)
        
        # Compute expert outputs
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Weighted combination
        output = torch.sum(
            routing_probs.unsqueeze(-1) * expert_outputs, dim=1
        )
        
        # Return output and routing probabilities for analysis
        return output, routing_probs


# Evaluation metrics
def compute_sparsity_metrics(routing_probs):
    """Compute sparsity metrics for routing probabilities"""
    # Average number of active experts (>1% probability)
    active_experts = (routing_probs > 0.01).float().sum(dim=1).mean()
    
    # Gini coefficient for concentration
    sorted_probs = routing_probs.sort(dim=1)[0]
    n = routing_probs.shape[1]
    index = torch.arange(1, n + 1).float().to(routing_probs.device)
    gini = (2 * (index * sorted_probs).sum(dim=1) / 
            (n * sorted_probs.sum(dim=1)) - (n + 1) / n).mean()
    
    return {
        "active_experts": active_experts.item(),
        "gini_coefficient": gini.item()
    }
