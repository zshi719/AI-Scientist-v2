# Game-Theoretic Expert Selection for Mixture-of-Experts

## Research Direction
This research explores expert selection in MoE architectures through a game-theoretic lens, modeling it as a two-player zero-sum game between a utility-maximizing Selector and an adversarial Nature.

## Key Innovations
1. **Endogenous Tsallis Entropy**: Regularization emerges naturally from the game equilibrium, not imposed ad-hoc
2. **Unified Framework**: Existing methods (softmax, sparsemax) are special cases of Nash equilibria
3. **Controllable Sparsity**: Parameter q provides post-training control over routing behavior

## Experimental Plan
- Implement game-theoretic routing with varying q parameters
- Test adversarial robustness under different uncertainty models
- Analyze convergence to Nash equilibrium
- Measure Price of Robustness across different settings
- Demonstrate post-training adaptability
