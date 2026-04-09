# Recurrent Dueling Double DQN for Exploitative Best Response

## Overview

A value-based exploitative agent for Heads-Up Limit Texas Hold'em. Unlike the adaptive actor-critic agent, this model is trained against one **fixed opponent archetype at a time** and is meant to learn a strong best response to that specific style.

**Hypothesis:** when the opponent is fixed and the action space is discrete, a dueling Double DQN can learn a profitable counter-strategy with a simpler training setup than policy-gradient methods.

## Architecture

```
              state features (90-dim)
                       │
               ┌───────▼───────┐
               │   FC 90→128   │
               └───────┬───────┘
                       │
               ┌───────▼───────┐
               │  FC 128→128   │
               └───────┬───────┘
                       │
               ┌───────▼───────┐
               │  Game LSTM    │  ← persists within one hand
               │   128 hidden   │
               └───────┬───────┘
                       │
             ┌─────────▼─────────┐
             │ Value head 128→1  │
             └─────────┬─────────┘
                       │
             ┌─────────▼─────────┐
             │Advantage 128→4    │
             └─────────┬─────────┘
                       │
          Q(s,a)=V(s)+(A(s,a)-mean_a A(s,a))
```

### Components

| Component | Parameters | Role | Lifecycle |
|---|---|---|---|
| Feedforward trunk | ~28k | Encodes RLCard observation, hand-strength features, and legal-action context | Runs every decision |
| Game LSTM | ~132k | Captures betting history within a hand | Resets every hand |
| Dueling heads | ~645 | Separates state value from action-specific advantage | Runs every decision |
| Target network | duplicate of online Q-network | Stabilizes temporal-difference targets | Synced periodically |

## Why This Design

### Why DQN at all?

This environment has a **small discrete action space**: `call`, `raise`, `fold`, `check`. That makes Q-learning a natural fit. We are also training against **fixed opponent policies**, which reduces the non-stationarity that often makes value-based RL unstable in multi-agent games.

### Why Double DQN?

Plain DQN tends to overestimate action values because the same network both selects and evaluates the bootstrap action. Double DQN reduces that bias:

1. The **online** network chooses the best next action.
2. The **target** network evaluates that chosen action.

That separation is especially useful here because poker rewards are sparse and noisy, so optimistic bias can be costly.

### Why dueling heads?

Many poker decisions are driven by two related questions:

1. how good is this state overall?
2. which legal action is best in this state?

A dueling network reflects that structure by learning a scalar **state value** `V(s)` and an **advantage** term `A(s, a)` for each action, then combining them into Q-values. That is a modest upgrade over a single linear Q-head and often helps when multiple actions are similar in value.

### Why a recurrent Q-network?

Poker is partially observable. The current public state does not fully reveal:

- the opponent's private cards
- the full latent trajectory that led here
- the strategic meaning of the betting sequence unless it is remembered

The LSTM gives the agent memory **within a hand**, so Q-values can depend on the action history rather than only the latest flat observation.

### Why only one LSTM, not an opponent model across hands?

This DQN is intentionally scoped as a **best-response learner**, not a general adaptive opponent identifier. Its job is:

1. pick one archetype
2. train against it for many hands
3. learn the exploit

So the recurrent state is only used for **within-hand partial observability**. Cross-hand opponent memory is left to the actor-critic approach.

## Input Features

The DQN now uses a DQN-specific 90-dimensional feature vector rather than the shared actor-critic feature path:

| Feature | Dimensions | Source |
|---|---|---|
| RLCard obs vector | 72 | Card one-hots + environment encoding |
| Hand rank one-hot | 10 | Engineered from hole cards + board |
| Draw flags | 3 | Flush draw, straight draw, boat draw |
| Legal action mask | 4 | Binary mask over legal actions |
| Player position | 1 | 0 = small blind, 1 = big blind |

These engineered features are intended to reduce how much raw poker hand semantics the DQN has to discover from scratch. The legal-action mask is part of the network input and is also used again when selecting actions so illegal moves are never chosen.

The actor-critic agent still uses the older shared 77-dimensional feature vector. This DQN-only split keeps the hand-strength experiment isolated to the value-based agent.

## Training Flow

### One hand of experience

The shared hand runner records a transition for each decision the agent makes:

1. observe state `s_t`
2. choose action `a_t` with epsilon-greedy policy
3. wait until the next time the agent acts, producing `s_{t+1}`
4. store reward `0` for intermediate transitions
5. at terminal, store the final payoff as the last reward

This means each poker hand becomes a **short sequence of agent decision points**, which is exactly what the recurrent Q-network trains on.

### Replay design

The replay buffer stores **whole hands**, not shuffled single transitions.

That choice was deliberate:

- the LSTM hidden state needs the decision sequence in order
- training on isolated timesteps would break the recurrence
- poker hands are short enough that full-sequence replay is practical

During an update, the agent samples several completed hands and unrolls the online and target networks across each one from a zero hidden state.

## Learning Rule

For each step in a sampled hand:

1. run the online network on `s_t` and select `Q(s_t, a_t)`
2. run the online network on `s_{t+1}` to choose the greedy next legal action
3. run the target network on `s_{t+1}` to evaluate that chosen action
4. build the Double DQN target

```
y_t = r_t + gamma * Q_target(s_{t+1}, argmax_a Q_online(s_{t+1}, a))
```

For terminal steps, the bootstrap term is zero.

The implementation uses **Huber loss** (`smooth_l1_loss`) rather than plain MSE because poker payoffs can be spiky and Huber is usually more forgiving to large TD errors early in training.

## Exploration Strategy

The policy is epsilon-greedy during training:

- start at `epsilon = 1.0`
- decay linearly toward `0.05`
- decay measured in **hands**, not optimizer steps

I chose hand-based decay because it matches the semantic unit of training in this project. One completed hand is a more intuitive measure of experience than one gradient update, especially since updates begin only after replay warmup.

## Stabilization Choices

### Target network

A frozen target network is updated every `target_update_every` optimization steps. This is standard DQN stabilization and is especially helpful here because:

- rewards arrive at the end of the hand
- training targets can otherwise chase a moving online network
- recurrent unrolling already adds enough complexity

### Warmup before learning

Training waits until the replay buffer has accumulated a minimum number of hands. This avoids immediately fitting on a tiny and highly correlated sample.

### Gradient clipping

Gradients are clipped before each optimizer step. Recurrent networks can produce unstable gradients, and clipping is a simple guardrail.

## Hyperparameters

Default settings in the current implementation:

| Parameter | Value |
|---|---|
| Learning rate | `1e-3` |
| Discount `gamma` | `1.0` |
| Epsilon start | `1.0` |
| Epsilon end | `0.05` |
| Epsilon decay | `20_000` hands |
| Replay capacity | `50_000` hands |
| Batch size | `32` hands |
| Warmup | `200` hands |
| Target update | every `200` optimizer steps |
| Gradient clip | `1.0` |

### Why `gamma = 1.0`?

Reward is only assigned at the end of the hand, and hands are short. Using `gamma = 1.0` keeps the target aligned with total hand EV rather than artificially discounting earlier betting decisions.

## Integration With This Repo

### Shared training loop

The DQN plugs into the same hand simulator used by the other agents. The key extension was making the hand runner emit **stepwise transitions** instead of only a terminal summary. That lets value-based learning work without creating a separate environment wrapper.

The shared hand runner now checks whether an agent provides its own feature builder. The DQN uses that hook to append engineered hand-strength features without changing the actor-critic training path.

### Opponent selection

Opponent creation was centralized in a shared registry so both actor-critic and DQN training can use the same archetype names.

### Evaluation

Session evaluation was generalized to load either:

- `ac:<model_name>`
- `dqn:<model_name>`

This keeps the plotting and benchmarking infrastructure shared across approaches.

## Design Tradeoffs

### What this implementation does well

- Fits naturally to the discrete action space
- Exploits fixed archetypes without needing opponent classification
- Uses recurrence where it matters most: within-hand betting history
- Uses dueling heads to separate state value from action preference
- Reuses the repo's existing training and evaluation infrastructure

### What it intentionally does not do

- It does **not** adapt across hands to infer an unknown opponent
- It does **not** use prioritized replay
- It does **not** use n-step returns or distributional RL
- It does **not** model belief state explicitly beyond recurrent hidden state

Those omissions were intentional to keep the first DQN version understandable, debuggable, and easy to compare against the actor-critic baseline.

## Current Limitations

There are a few important caveats in the current setup:

1. The replay buffer stores one scalar reward at terminal and zeros elsewhere, so credit assignment is still sparse.
2. The LSTM state is reset every hand, so this model learns a per-archetype exploit, not a session-adaptive exploit.
3. The replay samples full hands uniformly; it does not prioritize rare or high-error situations.
4. The training target is built from the next agent decision point, not every environment micro-step, which is a reasonable abstraction here but still a modeling choice.
5. The current engineered hand-strength features are intentionally coarse. They help with made-hand and draw recognition, but they do not fully encode kicker quality, blockers, or board texture.

## Future Improvements

If we want to push this further, the most promising next steps are:

1. Add prioritized episode replay.
2. Add n-step returns across the hand.
3. Add a distributional value head for a more Rainbow-like target.
4. Add an opponent-summary module across hands for a hybrid adaptive DQN.
5. Compare per-archetype DQN directly against CFR and AC in the same evaluation plots.
