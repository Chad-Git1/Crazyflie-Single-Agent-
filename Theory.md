# Actor-Critic Guassian policies

In a Gaussian (continous state space) policy, we sample actions a from a normal/Gaussian distribution N.
- a ~ N(μ, σ^2) = sample action a from normal distribution defined by mean μ and variance σ.

### The policy network (actor)
- Learns network weights & bias parameters, along with σ, all via gradient descent.
- μ (output of the policy NN) is also "learned" indirectly via the updated parameters. Paremeters change based on feedback from the critic, the actor learns to favour sampling high-return or unexplored actions.
- This network reads observations and learns a unique distribution (based on μ and σ) for each action a.

### The value network (critic)
- Reads observations and outputs V(s) ~ value estimate of current observation (state). 
- This value is used as feedback to the policy network to determine which actions are good/bad.

### Actor-Critic interaction
- For action a, more feedback may narrow σ as we become more confident about a's distribution, and may increase/decrease μ based on the estimated state value.
- This also balances exploration & exploitation since unkown actions will maintain a wide σ.

### The final policy
- Can just sample action with best mean (deterministic) or maintain some exploration with minor deviation σ.


# PPO-Clip Actor-Critic Guassian policy

### Same Principles as Other Actor-Critic Gaussian Policies
- Policy network parameters determine the distribution for each action a ~ N(μ, σ^2)
  - Weights and bias (predict μ)
  - log(σ)
- State value network performs as usual
  - Weight and bias parameters

PPO-Clip builds on regular policy gradient by clipping gradient steps
- Takes minimum step of (step, CLIP(step, 1 - 𝜖, 1 + 𝜖))
- Also performs minibatching over transition data set to mitigate ram usage