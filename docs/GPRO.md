# GPRO: Generalized Policy Optimization

**GPRO (Generalized Policy Optimization)** is the optimization algorithm that trains the Actor-Critic in the hybrid system. It updates both the Actor (policy) and the Critic (value estimator) using the stream of (query, response, reward) data from human feedback and the composite reward.

---

## Role in the framework

- **Actor (θ_actor):** The policy that selects which response to show (and optionally which temperature to use). GPRO updates the Actor to maximize expected reward (e.g. policy gradient, clipped objective, or group-normalized advantages).
- **Critic (θ_critic):** The value estimator that predicts expected score for (prompt, query, response). GPRO updates the Critic to minimize prediction error (e.g. MSE on the observed reward).

The training loop collects transitions (query, candidates, chosen index, reward) and every **M** queries (or every episode) calls the GPRO optimizer to perform one or more update steps on the Actor and the Critic.

---

## Typical update pattern

1. **Collect a batch** over M queries: for each query, store (query, list of candidate responses, index chosen by Actor, reward). Reward is typically the composite R_total (human feedback, Critic score, coherence, etc.) or human_feedback only.
2. **Critic update:** For each (prompt, query, response_shown, reward) in the batch, update the Critic (e.g. gradient step to minimize (Critic_θ(prompt, query, response) - reward)²).
3. **Actor update:** Use the batch to compute advantages (e.g. reward - baseline, or group-normalized reward within the batch) and update the Actor (e.g. policy gradient or clipped surrogate objective with KL penalty).

Advantage can be:

- **Critic-based:** A = reward - Critic_θ(prompt, query, response) (or discounted return - V).
- **Group-normalized (GPRO-style):** For each query, multiple candidates get rewards; A_i = (r_i - r̄) / √(Var(r) + ε) where r̄ is the mean reward over the group of responses for that query. This avoids learning a separate value function for advantage estimation.

---

## Interface in the framework

The framework provides an abstract **GPROOptimizer** in `prompt_rl.training.gpro`:

- **`update(batch, actor, critic)`** (or **`update_actor(batch, actor)`** and **`update_critic(batch, critic)`**): Performs one GPRO update step on the given batch. The batch is a list of transition objects (query, candidates, chosen_index, reward, and optionally prompt, response_shown).

You can plug in your own GPRO implementation (e.g. PPO-style Actor + MSE Critic, or group-normalized advantages for the Actor and MSE for the Critic) by implementing this interface and passing it to the training loop (e.g. when M queries are reached, call `gpro_optimizer.update(recent_batch, flow.actor, flow.critic)`).

---

## Summary

| Component | GPRO role |
|-----------|-----------|
| **Actor** | Updated by GPRO to maximize expected reward (policy gradient / clipped / group-normalized advantage). |
| **Critic** | Updated by GPRO to minimize value prediction error (e.g. MSE on reward). |
| **Batch** | (query, candidates, chosen_index, reward) per step; collected every M queries. |
| **Training loop** | Every M queries, passes the recent batch to GPROOptimizer.update(...). |

GPRO is the single optimization algorithm that trains both the Actor and the Critic; the evolutionary component (population, fitness, mutation, crossover) runs separately every E episodes and does not use GPRO.
