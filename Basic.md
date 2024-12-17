# BASICS
## Concept
**State** $s$ is a complete description of the state of the world.  
**Observation** $o$ is a partial description of a state, which may omit information.  
**Policy** $\pi$ is a rule used by an agent to decide what actions to take. Policy can be deterministic (确定性的) $a_t=\mu(s_t)$ or stochastic (随机的) $a_t \sim \pi(\cdot \mid s_t)$.  
**Trajectory** $\tau$ is a sequence of states and actions, $\tau = (s_0, a_0, s_1, a_1, \dots)$, where $s_0 \sim \rho_0(\cdot)$ is the initial state, state transitions are $s_{t+1} = f(s_t, a_t)$ (deterministic) or $s_{t+1} \sim P(\cdot|s_t, a_t)$ (stochastic), and actions are chosen by the agent's policy.  
**Reward** $r_t$ is a scalar feedback signal given to the agent at each timestep $t$, based on the current state and action pair $(s_t, a_t)$.  
**Return** $R_t$ is the cumulative discounted reward from timestep $t$ onward, defined as $R_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$, where $\gamma \in [0, 1]$ is the **discount factor**. 

## RL Problem Formulation
The goal in RL is to find a policy $\pi$ that **maximizes the expected return** $J(\pi)$, where the return can be finite-horizon or infinite-horizon.  
For stochastic environments and policies, the probability of a trajectory $\tau$ is: $P(\tau|\pi) = \rho_0(s_0) \prod_{t=0}^{T-1} P(s_{t+1} | s_t, a_t) \pi(a_t | s_t).$  
The expected return is: $J(\pi) = E_{\tau \sim \pi}[R(\tau)]$  
The central optimization problem is: $\pi^* = \text{arg max}_{\pi} J(\pi)$ where $\pi^*$ is the **optimal policy**.

## Notes
### Categorical and Diagonal Gaussian Policies

#### **Categorical Policies**  
- **Action Space**: Discrete  
- **Network Output**: Action probabilities via **softmax**.  
- **Sampling**: Use `torch.multinomial` or `torch.distributions.Categorical`.  
- **Log-Likelihood**: $log π_θ(a|s) = log [P_θ(s)]_a$ Where $P_θ(s)$ is the probability vector.

#### **Diagonal Gaussian Policies**  
- **Action Space**: Continuous  
- **Network Output**: Mean $μ_θ(s)$ and log standard deviation $log σ$.  
  - **Fixed**: $log σ$ is standalone.  
  - **State-Dependent**: $log σ_θ(s)$.  
- **Sampling**:  $a = μ_θ(s) + σ_θ(s) ⊙ z$, where $z ~ N(0, I)$  
- **Log-Likelihood**:  $log π_θ(a|s) = -1/2 * [ Σ_i^k ( (a_i - μ_i)^2 / σ_i^2 + 2 log σ_i ) + k log 2π ]$
