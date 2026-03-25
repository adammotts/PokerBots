# Adaptive Actor-Critic with Opponent Modeling

## Overview

A general-purpose exploitative agent for Heads-Up Limit Texas Hold'em. The agent does **not** know which opponent it faces. It adapts its policy over the course of many hands by observing opponent behavior, identifying tendencies, and exploiting them.

**Hypothesis:** Performance similar to CFR early in a session (when the opponent is unknown) and superior later (once the opponent model has accumulated enough evidence to exploit).

## Architecture

```
                    ┌─────────────────┐
                    │  Opponent LSTM   │  ← persists across hands
                    │  (8→32 hidden)   │
                    └────────┬────────┘
                             │ opp_context (32-dim)
                    ┌────────┴────────┐
                    │                 │
            ┌───────▼───────┐ ┌──────▼───────┐
            │     Actor     │ │    Critic    │
            │  FC(77→128→64)│ │ FC(77→128→64)│
            │  Game LSTM(64)│ │ Game LSTM(64)│  ← reset each hand
            │  Head(96→32→4)│ │ Head(96→32→1)│
            └───────┬───────┘ └──────┬───────┘
                    │                │
              action logits    state value V(s)
```

### Components

| Component | Parameters | Role | Lifecycle |
|---|---|---|---|
| Actor (FC + Game LSTM + Head) | ~25k | Policy: state → action logits | Game LSTM resets each hand |
| Critic (FC + Game LSTM + Head) | ~25k | Value function: state → V(s) | Game LSTM resets each hand |
| Opponent LSTM | ~5k | Cross-hand opponent model | Persists across hands in a session |
| Confidence Gate | ~33 | Learned confidence for KL gating | Only used in KL variant |

Actor and critic are **separate networks** (no shared backbone) to avoid policy gradient updates corrupting value estimates. The opponent LSTM is shared — both actor and critic read from its hidden state.

### Game LSTM vs Opponent LSTM

These are two different LSTMs with fundamentally different roles:

- **Game LSTM** (inside actor and critic): Processes the sequence of actions *within a single hand* (preflop → flop → turn → river). Captures partial observability within one deal. **Resets every hand.**

- **Opponent LSTM** (standalone): Processes a *summary of each completed hand* as a time series across many hands. After hand 1, it gets "opponent called 3x, raised 1x, went to showdown, I lost 10 chips." By hand 50, its hidden state encodes "this opponent calls everything and never folds." **Never resets during a session.** This is what enables exploitation.

## Input Features

### Per decision point (77-dim)

| Feature | Dimensions | Source |
|---|---|---|
| RLCard obs vector | 72 | Card one-hots (52) + raise count encoding (20) |
| Legal action mask | 4 | Binary over {call, raise, fold, check} |
| Player position | 1 | 0 = small blind, 1 = big blind |

### Opponent hand summary (8-dim, fed to Opponent LSTM after each hand)

| Feature | Dimensions | Description |
|---|---|---|
| Opponent action frequencies | 4 | Normalized call/raise/fold/check counts |
| Showdown result | 1 | +1 win, -1 loss, 0 no showdown |
| Went to showdown | 1 | Binary flag |
| Payoff normalized | 1 | payoff / big_blind |
| Rounds reached | 1 | 1-4 (preflop through river) |

## Training Algorithm (A2C with Monte Carlo Returns)

### Per-hand update

1. **Reset** game LSTM hidden states (actor + critic). Keep opponent LSTM hidden.
2. **Play hand**: at each decision point, forward through actor (get logits) and critic (get value). Mask illegal actions, sample action from Categorical distribution. Store trajectory.
3. **End of hand**: get payoff R.
4. **Compute returns**: G_t = R for all timesteps (γ=1.0, reward only at terminal).
5. **Compute advantages**: A_t = R - V(s_t) (Monte Carlo advantage).
6. **Compute loss**:
   - `policy_loss = -mean(log_prob_t * A_t.detach())`
   - `value_loss = MSE(V(s_t), R)`
   - `entropy_bonus = -mean(entropy_t)`
   - `total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus`
   - If KL variant: `+ lambda_kl * KL(agent_probs || cfr_probs)`
7. **Backprop**: `optimizer.zero_grad()` → `loss.backward()` → `clip_grad_norm_(0.5)` → `optimizer.step()`
8. **Step opponent LSTM**: build hand summary, feed through opponent LSTM with `torch.no_grad()`.

### Why Monte Carlo returns (not TD)?

Poker hands are short (2-8 decision points) and all reward comes at the terminal state. Full-episode returns are practical and avoid the complexity of n-step TD in a partially observable domain.

## Adaptive KL Regularization (Novel Contribution)

**Training-time only.** At inference, the agent is a pure neural network forward pass — no CFR involved.

```
cfr_probs = frozen_cfr.get_action_probs(state)       # query during training
agent_log_probs = log_softmax(masked_logits)
kl = KL(agent_log_probs || cfr_probs)                # divergence from Nash

confidence = sigmoid(confidence_gate(opp_hidden))     # 0 early, ~1 later
lambda_kl = lambda_max * (1 - confidence)             # strong early, weak later

total_loss += lambda_kl * kl
```

**How it works**: The confidence gate is a small linear layer on the opponent LSTM's hidden state. When the opponent LSTM has seen few hands, its hidden state is near-zero → confidence ≈ 0 → strong KL penalty → agent plays near-Nash. As the opponent LSTM accumulates evidence, the hidden state becomes more structured → confidence rises → KL penalty fades → agent exploits.

**Why this is novel**: Existing work uses either fixed KL regularization (NashPG) or fixed policy mixing (safe exploitation). No prior work ties the regularization strength to a *learned opponent-model confidence signal*.

**Inference behavior**: The KL penalty shapes the weights during training such that the agent's natural behavior (without any CFR query) starts Nash-like with a fresh opponent LSTM state and becomes exploitative as the opponent LSTM state evolves.

## Training Setup

### Episode structure

One episode = 1,000 hands against a randomly-selected opponent archetype. At the start of each episode, the opponent LSTM hidden state resets (fresh slate for new opponent). The archetype is randomly chosen from {Calling Station, Maniac, Old Man Coffee}.

### Why random mixture (not per-archetype)?

The agent must learn to *identify* which opponent it faces, not just *exploit* a known opponent. If trained against one archetype, the opponent LSTM would never learn to distinguish anything. The random mixture forces the opponent LSTM to develop discriminative representations.

### Hyperparameters

| Parameter | Value |
|---|---|
| Episodes | 50 (each = 1,000 hands) |
| Learning rate | 3e-4 (Adam) |
| Discount γ | 1.0 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Gradient clip | 0.5 |
| KL λ_max | 0.0 (pure) or 0.5 (KL variant) |

### Two variants (A/B comparison)

| Variant | λ_max | Description |
|---|---|---|
| `ac_pure` | 0.0 | Pure A2C, no Nash prior |
| `ac_kl` | 0.5 | A2C + adaptive KL toward CFR |

## Evaluation

Evaluated per-archetype against CFR baseline using the existing `Evaluator` class. Key metrics:
- **mbb/h** (milli-big-blinds per hand) overall
- **Early vs late mbb/h**: hands 1-100 vs hands 900-1000 within a session, showing adaptation
- **Cross-archetype**: same model tested against all archetypes to show generalization
