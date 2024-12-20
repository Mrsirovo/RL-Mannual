# Trust Region Policy Optimization (TRPO)

## Background

TRPO is an on-policy algorithm designed to improve policy performance by taking the largest possible step while ensuring the new policy remains close to the old one. This closeness is measured using the Kullback-Leibler (KL) divergence, which quantifies the difference between two probability distributions. Unlike standard policy gradient methods that constrain updates in parameter space—where small parameter changes can lead to significant performance variations—TRPO's approach helps prevent performance collapse, allowing for larger, safer updates and enhancing sample efficiency.

## Key Equations

Given a policy parameterized by $\theta$, denoted as $\pi_{\theta}$, the theoretical TRPO update is defined as:

$$
\theta_{k+1} = \arg \max_{\theta} \; \mathcal{L}(\theta_k, \theta) \quad \text{s.t.} \quad D_{KL}(\theta \| \theta_k) \leq \delta
$$

Here:

- $L(\theta_k, \theta)$ represents the surrogate advantage, assessing how the new policy $\pi_{\theta}$ performs relative to the old policy $\pi_{\theta_k}$ using data from $\pi_{\theta_k}$ :

$$
L(\theta_k, \theta) = E_{s,a \sim \pi_{\theta_k}} \left[ \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a) \right]
$$

- $D_{KL} (\theta \| \theta_k)$ is the average KL divergence between the new and old policies across states visited by $\pi_{\theta_k}$ :

$$
D_{KL}(\theta \| \theta_k) = E_{s \sim \pi_{\theta_k}} \left[ D_{KL}\left(\pi_{\theta}(\cdot|s) \| \pi_{\theta_k}(\cdot|s) \right) \right]
$$

To facilitate computation, TRPO approximates these expressions using first-order Taylor expansions around $\theta_k$ :

$$
L(\theta_k, \theta) \approx g^T (\theta - \theta_k)
$$

$$
D_{KL}(\theta \| \theta_k) \approx \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k)
$$

where $g$ is the gradient of the surrogate advantage function with respect to $\theta$, and $H$ is the Hessian of the average KL divergence. This leads to the simplified optimization problem:

$$
\theta_{k+1} = \arg \max_{\theta} \; g^T (\theta - \theta_k) \quad \text{s.t.} \quad \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k) \leq \delta
$$

## Pseudocode

A high-level overview of the TRPO algorithm is as follows:

1. **Initialize** policy parameters $\theta_0$.
2. **Repeat** for each iteration $k = 0, 1, 2, \ldots$:
   - **Collect** a set of trajectories $\mathcal{D}$ by executing the current policy $\pi_{\theta_k}$.
   - **Compute** advantage estimates $A^{\pi_{\theta_k}}(s,a)$ using the collected data.
   - **Estimate** the gradient $g$ of the surrogate advantage function.
   - **Estimate** the Hessian $H$ of the average KL divergence.
   - **Solve** the constrained optimization problem to obtain the new policy parameters $\theta_{k+1}$.
   - **Update** the policy: $\theta_k \leftarrow \theta_{k+1}$.
