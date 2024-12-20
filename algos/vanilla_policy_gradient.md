# Vanilla Policy Gradient (VPG)

## Background

The Vanilla Policy Gradient (VPG) algorithm is a foundational approach in reinforcement learning that optimizes policy parameters through gradient ascent on expected returns. It is an on-policy method applicable to both discrete and continuous action spaces.

## Key Equations

Let $\pi_{\theta}$ denote a policy with parameters $\theta$, and $J(\pi_{\theta})$ represent the expected return. The policy gradient is given by:

$$
\nabla_{\theta} J(\pi_{\theta}) =E_{\tau \sim \pi_{\theta}} \left[ \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) A^{\pi_{\theta}}(s_t,a_t) \right]
$$

where $\tau$ is a trajectory, and $A^{\pi_{\theta}}$ is the advantage function under the current policy.

The policy parameters are updated via stochastic gradient ascent:

$$
\theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta_k})
$$

## Exploration vs. Exploitation

VPG trains a stochastic policy in an on-policy way. This means that it explores by sampling actions according to the latest version of its stochastic policy. The amount of randomness in action selection depends on both initial conditions and the training procedure. Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the policy to get trapped in local optima.

## Algorithm

Below is the **Vanilla Policy Gradient Algorithm** wrapped in a code block for clarity:

```python
Algorithm: Vanilla Policy Gradient
Input: Initial policy parameters θ₀, initial value function parameters φ₀
for k = 0, 1, 2, ... do
    # 1. Collect trajectories Dₖ by running policy πₖ = π(θₖ) in the environment
    Collect Dₖ = {τᵢ} by running πₖ = π(θₖ)

    # 2. Compute rewards-to-go for each trajectory
    Compute rewards-to-go: R̂ₜ

    # 3. Compute advantage estimates using the current value function V_{φₖ}
    Compute advantage estimates: Âₜ based on V_{φₖ}

    # 4. Estimate policy gradient
    ĝₖ = (1 / |Dₖ|) * Σ_{τ ∈ Dₖ} Σ_{t=0}^T ∇_θ log πₖ(aₜ | sₜ) Âₜ

    # 5. Update policy parameters
    θₖ₊₁ = θₖ + αₖ * ĝₖ

    # 6. Fit value function by minimizing mean-squared error
    φₖ₊₁ = argmin_φ (1 / |Dₖ|T) Σ_{τ ∈ Dₖ} Σ_{t=0}^T (V_φ(sₜ) - R̂ₜ)²
end for
