# Actor-Critic Architecture Evolution

## Problem Statement

Build a Heads-Up Limit Texas Hold'em agent that **adapts its strategy within a session** to exploit specific opponent tendencies. The agent should match CFR (Nash equilibrium) performance early in a session and surpass it once the opponent's patterns become identifiable.

**Environment**: RLCard limit-holdem. 72-dim observation (52 card one-hots + 20 raise count features). 4 discrete actions: call, raise, fold, check. Reward only at terminal state (hand payoff). Positions randomly assigned each hand.

---

## Architecture 1: Dual-LSTM with Concatenation

**Commits**: `87c2804` through `1475944`
**Saved results**: `PPO_32_roundrobin`, `PPO_128_wrandom`, `PPO_128_wrandom_aux`

### Motivation

Inspired by He et al. (DRON, ICML 2016) and the entropy-regularized actor-critic paper (doi:10.3390/e24060774). The hypothesis: a recurrent opponent model that persists across hands can learn to identify opponent archetypes and condition the policy accordingly.

### Architecture

Two separate LSTM systems operating at different timescales:

```
PER DECISION POINT:

  game_state (77-dim)
       │
       ▼
  FC(77→128) → ReLU → FC(128→64) → ReLU
       │
       ▼
  Game LSTM(64→64)          ← resets each hand
       │
       ├── opp_context (32-dim) from Opponent LSTM
       │
       ▼
  Concatenate [game_lstm_out(64), opp_context(32)] = 96-dim
       │
       ▼
  FC(96→32) → ReLU → FC(32→4) → action logits


BETWEEN HANDS:

  hand_summary (8-dim)
       │
       ▼
  Opponent LSTM(8→32)       ← persists across hands within session
       │
       ▼
  opp_context (32-dim)      → fed to actor and critic at next hand
```

**Actor pseudocode**:
```python
class ActorNetwork:
    trunk:      Linear(77→128), ReLU, Linear(128→64), ReLU
    game_lstm:  LSTM(input=64, hidden=64)
    head:       Linear(96→32), ReLU, Linear(32→4)

    def forward(features, game_hidden, opp_context):
        x = trunk(features)                          # (1, 64)
        x, game_hidden = game_lstm(x, game_hidden)   # (1, 64)
        x = concat([x, opp_context])                 # (1, 96)
        logits = head(x)                             # (1, 4)
        return logits, game_hidden
```

**Critic**: Identical structure to actor but outputs scalar V(s) instead of 4-dim logits.

**Opponent LSTM** (shared between actor and critic):
```python
class OpponentLSTM:
    lstm: LSTM(input=8, hidden=32)

    def forward(hand_summary, hidden):
        # hand_summary: [call_freq, raise_freq, fold_freq, check_freq,
        #                showdown_result, went_to_showdown, payoff/10, rounds_norm]
        output, hidden_new = lstm(hand_summary, hidden)
        return output, hidden_new
```

**Opponent hand summary (8-dim)**:
| Index | Feature | Range |
|-------|---------|-------|
| 0-3 | Opponent action frequencies (call, raise, fold, check) | [0, 1] normalized |
| 4 | Showdown result | -1, 0, or +1 |
| 5 | Went to showdown | 0 or 1 |
| 6 | Payoff / 10 | continuous |
| 7 | Betting rounds reached (normalized) | [0.25, 1.0] |

**Parameters**: Actor ~25K, Critic ~25K, Opponent LSTM ~5K, total ~55K

### Training Variants

**PPO_32_roundrobin** (150 episodes):
- A2C initially, then PPO with rollout_size=32
- Round-robin opponent selection across 4 hardcoded opponents
- Per-hand updates, Monte Carlo returns (γ=1.0)

**PPO_128_wrandom** (150 episodes):
- PPO with rollout_size=128
- Softmax-weighted random opponent selection (inverse count)
- Learning rate decay: `lr * max(1 - ep/total, 0.05)`
- Entropy coefficient decay: `0.01 * max(1 - ep/total, 0.1)`

**PPO_128_wrandom_aux** (380 episodes):
- Added auxiliary opponent prediction loss (MSE on cumulative action frequencies)
- `aux_coef=0.5`, OpponentPredictor: Linear(32→16) → ReLU → Linear(16→4) → Softmax

### Results and Diagnosis

All variants converged to ~0 average payoff per hand. Per-opponent: old_man_coffee ~+0.5 (trivial), calling_station ~0, maniac ~0, polarizing ~0. **No within-session adaptation** — first-half and second-half payoffs were identical.

**Root causes identified**:
1. **Card-luck variance drowned gradient signal**: ~100× noise-to-signal ratio. The opponent LSTM received no useful gradient because poker payoffs are dominated by card deals, not strategy quality.
2. **Game LSTM was redundant**: RLCard's 72-dim observation already encodes raise counts per betting round. The within-hand LSTM re-learned what was already in the features.
3. **Concatenation allowed the actor to ignore opp_context**: The actor could learn a good policy using only the 64-dim game LSTM output, treating the 32-dim opp_context as noise.
4. **Only 4 deterministic opponents**: Insufficient diversity for the opponent model to learn generalizable representations.

---

## Architecture 2: CTDE with FiLM Conditioning

**Commits**: `69decc2` (CTDE refactor), `43e31e1` (parameterized opponents)
**Saved results**: `CTDE_512_wrandom_aux`, `CTDE_512_param_aux`, `CTDE_512_mixed_aux`

### Motivation

Addressed each diagnosed failure from Architecture 1:
- **Card variance** → Centralized critic sees both hands (Srinivasan et al., NeurIPS 2018)
- **Redundant game LSTM** → Feedforward actor (as used by NFSP, DREAM, DeepStack, AlphaHoldem)
- **Concatenation bypass** → FiLM conditioning (Perez et al., AAAI 2018; used in AlphaStar, Vinyals et al., Nature 2019)
- **Weak opponent supervision** → Cross-entropy aux loss on actual opponent actions (He et al., DRON, ICML 2016; Rabinowitz et al., ToMnet, ICML 2018)
- **Low diversity** → Parameterized opponents with continuous parameters

### Architecture

```
PER DECISION POINT:

  game_state (77-dim)
       │
       ▼
  MLP Trunk: Linear(77→128) → ReLU → Linear(128→128) → ReLU
       │
       │     opp_context (32-dim) from Opponent LSTM
       │          │
       │     ┌────┴────┐
       │     │  FiLM   │
       │     │ γ=Linear(32→128)
       │     │ β=Linear(32→128)
       │     └────┬────┘
       │          │
       ▼          ▼
  x = γ ⊙ trunk_out + β         ← element-wise modulation
       │
       ▼
  Head: Linear(128→64) → ReLU → Linear(64→4) → action logits


CENTRALIZED CRITIC (training only):

  game_state (77-dim) + both_hands_onehot (104-dim)
       │
       ▼
  Concatenate → (181-dim)
       │
       ▼
  MLP Trunk: Linear(181→128) → ReLU → Linear(128→128) → ReLU
       │
       │     opp_context (32-dim) ──► FiLM (same mechanism)
       ▼
  Head: Linear(128→64) → ReLU → Linear(64→1) → V(s)


BETWEEN HANDS (unchanged from Architecture 1):

  hand_summary (8-dim) → Opponent LSTM(8→32) → opp_context


AUXILIARY LOSS:

  opp_context (32-dim) → OpponentPredictor → 4-dim logits
       │
       ▼
  Cross-entropy loss vs opponent's actual actions this hand
```

**Actor pseudocode**:
```python
class ActorNetwork:
    trunk:      Linear(77→128), ReLU, Linear(128→128), ReLU
    film_scale: Linear(32→128)
    film_shift: Linear(32→128)
    head:       Linear(128→64), ReLU, Linear(64→4)

    def forward(features, opp_context):
        x = trunk(features)                    # (B, 128)
        gamma = film_scale(opp_context)         # (B, 128)
        beta = film_shift(opp_context)          # (B, 128)
        x = gamma * x + beta                   # FiLM modulation
        logits = head(x)                       # (B, 4)
        return logits
```

**Centralized critic pseudocode**:
```python
class CentralizedCritic:
    trunk:      Linear(181→128), ReLU, Linear(128→128), ReLU
    film_scale: Linear(32→128)
    film_shift: Linear(32→128)
    head:       Linear(128→64), ReLU, Linear(64→1)

    def forward(features, both_hands_onehot, opp_context):
        x = trunk(concat([features, both_hands_onehot]))  # (B, 128)
        gamma = film_scale(opp_context)                     # (B, 128)
        beta = film_shift(opp_context)                      # (B, 128)
        x = gamma * x + beta
        value = head(x)                                     # (B, 1)
        return value
```

**Card encoding** for centralized critic:
```python
def encode_both_hands_onehot(hand_0, hand_1):
    # Uses RLCard card2index.json mapping
    # Card format: suit + rank, e.g. "SA" = Spade Ace
    vec = zeros(104)
    for card in hand_0: vec[card2index[card.suit + card.rank]] = 1.0
    for card in hand_1: vec[52 + card2index[card.suit + card.rank]] = 1.0
    return vec  # (104,)
```

**Key changes from Architecture 1**:
- Actor is fully feedforward (no game LSTM) → all steps batchable, 2-3× speedup
- FiLM conditioning replaces concatenation → opponent context modulates features multiplicatively
- Centralized critic sees both players' hole cards → eliminates card-luck variance from advantages
- GAE (λ=0.95) replaces Monte Carlo returns (Schulman et al., ICLR 2016)
- Cross-entropy aux loss on individual opponent actions replaces MSE on cumulative frequencies
- Two separate optimizers: actor (+ opponent LSTM + predictor) and critic
- 5 extra critic-only gradient steps per PPO epoch (Srinivasan et al., 2018)
- High entropy coefficient 0.05 (Rudolph et al., ICLR 2025: PPO with high entropy outperforms NaD/PSRO in imperfect-info games)

**Parameters**: Actor ~43K, Critic ~65K, Opponent LSTM ~5K, FiLM ~8K, Predictor ~1K, total ~107K

### PPO Update with GAE

```python
def ppo_update(rollout):
    for epoch in range(4):
        opp_hidden = opp_hidden_rollout_start.clone()

        for hand in rollout:
            opp_context = opp_hidden[0].squeeze(0)     # (1, 32)

            # Batched forward (feedforward actor, no LSTM)
            logits = actor(step_features, opp_context.expand(n_steps, -1))
            values = critic(step_features, both_hands, opp_context.expand(n_steps, -1))

            # GAE per hand
            for t in reversed(range(T)):
                delta = reward[t] + gamma * V[t+1] - V[t]
                gae[t] = delta + gamma * lambda * gae[t+1]
            advantages = gae
            returns = gae + values.detach()

            # Aux loss: predict opponent actions from opp_context BEFORE this hand
            pred = opp_predictor(opp_context)
            aux_loss += cross_entropy(pred, hand.opp_actions)

            # Step opponent LSTM (with gradients in meta mode)
            _, opp_hidden = opponent_lstm(hand_summary, opp_hidden)

        # PPO clipped surrogate
        ratio = exp(log_prob_new - log_prob_old)
        surr = min(ratio * adv, clamp(ratio, 0.8, 1.2) * adv)
        policy_loss = -mean(surr)
        value_loss = MSE(values, returns)
        total_loss = policy_loss + 0.05 * entropy_loss + 0.3 * aux_loss + 0.5 * value_loss
        backward(total_loss)
```

### Training Variants

**CTDE_512_wrandom_aux** (150ep, 200ep):
- 4 hardcoded opponents only, softmax weighted random selection
- rollout_size=512, constant LR (no decay)
- First result showing clear improvement: overall +0.4, maniac from -3.0 to ~0

**CTDE_512_param_aux** (150ep, 200ep):
- Parameterized opponents only (balanced across 4 categories)
- `ParameterizedPlayer(vpip, pfr, aggression, fold_to_raise)` with hand strength heuristic
- Categories: tight-passive, tight-aggro, loose-passive, loose-aggro
- Training payoff up to +1.5, but validation against hardcoded opponents oscillated

**CTDE_512_mixed_aux** (150ep, 390ep):
- 50/50 mix: hardcoded + parameterized opponents
- Stabilized validation: OMC +0.65, maniac ~0, calling_station ~0, polarizing ~0
- Periodic validation every 5 episodes against hardcoded 4 (plotted as 4th panel)

### Results and Diagnosis

Strong base policy improvement over Architecture 1. Overall payoff reached +0.4 (from 0). Maniac went from -3.0 to breakeven. Old_man_coffee stable at +0.6. **But adaptation remained flat** — first-half ≈ second-half across all variants.

**Diagnosis**: FiLM learned near-identity transforms (γ≈1, β≈0). The actor found a single static policy that performed reasonably against all opponents and had no gradient pressure to use opp_context differently. The opponent LSTM hidden state evolved across hands but the actor didn't respond to those changes.

**Key insight**: The problem was not the architecture — it was the **training objective**. PPO optimizes per-hand rewards. An agent that adapts at hand 10 vs one that plays a fixed good strategy from hand 1 achieve similar per-hand rewards. No explicit incentive to adapt fast.

---

## Architecture 2b: Meta-Batched Training (Training Loop Change)

**Commits**: `47517fc` (meta training), `995e877` (plotting fix)
**Saved results**: `ac_v2_meta` (500 meta-iterations)

### Motivation

The RL² paper (Duan et al., 2016) identified that adaptation emerges when the training objective optimizes for **cumulative trial return** with the recurrent hidden state in the computation graph. Our PPO was detaching the opponent LSTM at rollout boundaries, breaking the gradient path from late-hand rewards to early-hand opponent representations.

### Architecture

**Same networks as Architecture 2** (CTDE/FiLM). Only the training loop changes.

### Training Loop Change

**Previous loop** (Architecture 2):
```
for episode in range(num_episodes):
    opponent = sample_opponent()
    agent.reset_opponent_state()
    while hand < 500:
        batch = min(512, remaining_hands)
        agent.begin_collect()
        for _ in range(batch):
            play_hand()
            agent.finish_hand_collect()
        agent.ppo_update()        # opp LSTM detached at rollout boundaries
```

**Meta-batched loop**:
```
for meta_iter in range(num_meta_iters):
    all_trials = []
    for trial in range(8):                    # 8 trials per update
        opponent = sample_opponent()           # fresh opponent each trial
        agent.reset_opponent_state()           # LSTM hidden → zeros
        agent.begin_collect()
        for hand in range(100):               # 100 hands per trial
            play_hand()
            agent.finish_hand_collect()
        all_trials.append(agent.get_trial())

    agent.meta_ppo_update(all_trials)          # LSTM IN computation graph
```

**Key difference in `meta_ppo_update`**:
```python
def meta_ppo_update(trials):
    for epoch in range(ppo_epochs):
        for trial in trials:
            opp_hidden = zeros()               # fresh per trial

            for hand in trial:
                opp_context = opp_hidden[0]    # IN the graph (not detached!)

                logits = actor(features, opp_context)
                values = critic(features, both_hands, opp_context)
                # ... GAE, aux loss ...

                _, opp_hidden = opponent_lstm(summary, opp_hidden)
                # opp_hidden stays in graph!
                # Gradient flows: hand_80 advantage → opp_context → LSTM → hand_1..79 summaries

        # Single backward() through all trials and all LSTM steps
        total_loss.backward()
```

The opponent LSTM processes ~100 hand summaries per trial (8→32 LSTM, very cheap). Actor/critic forward passes remain batched (feedforward). Total sequential LSTM steps: ~800 per PPO epoch (8 trials × 100 hands).

### Results

Base policy comparable to Architecture 2 (overall +1.0, OMC +0.65, calling_station ~0). **Adaptation still flat** — green and red lines overlapping. Maniac validation volatile (-4 to +3).

The meta-batching put the LSTM in the graph but the gradient through 100 LSTM steps may have been too weak to overcome the dominant per-hand policy gradient signal. The LSTM is only 5K parameters receiving gradient from ~800 steps, while the actor (43K params) receives gradient from ~2400 action steps. The adaptation signal is diluted.

---

## Architecture 3: RL² Session LSTM

**Saved results**: `ac_v3` (500 episodes)

### Motivation

If the separate opponent LSTM is too easy for the actor to ignore (even with FiLM, even with meta-batching), put the LSTM **inside the actor's forward path** where it cannot be bypassed. This is the pure RL² approach (Duan et al., 2016): the policy network is itself recurrent, processing augmented inputs that include previous actions and rewards.

### Architecture

```
PER DECISION POINT:

  [game_state(77), prev_action_onehot(4), prev_reward(1), hand_done(1)] = 83-dim
       │
       ▼
  MLP Encoder: Linear(83→128) → ReLU → Linear(128→128) → ReLU
       │
       ▼
  Session LSTM(128→64)       ← persists across ALL hands in session
       │
       ▼
  Head: Linear(64→32) → ReLU → Linear(32→4) → action logits


CENTRALIZED CRITIC:
  game_state(77) + both_hands(104)
       │
       ▼
  Trunk(181→128) → FiLM from session_hidden(64) → Head → V(s)
```

**Actor pseudocode**:
```python
class ActorNetwork:
    encoder:      Linear(83→128), ReLU, Linear(128→128), ReLU
    session_lstm: LSTM(input=128, hidden=64)
    head:         Linear(64→32), ReLU, Linear(32→4)

    def forward(features_aug, session_hidden):
        x = encoder(features_aug)                              # (1, 128)
        x, session_hidden_new = session_lstm(x, session_hidden) # (1, 64)
        logits = head(x)                                       # (1, 4)
        return logits, session_hidden_new
```

**Augmented features**:
```python
def build_augmented_features(state, prev_action, prev_reward, hand_done):
    base = build_features(state)                    # (77,)
    prev_act_onehot = one_hot(prev_action, 4)       # (4,)
    prev_rew = [prev_reward / 10.0]                 # (1,)
    hd = [1.0 if hand_done else 0.0]                # (1,)
    return concat([base, prev_act_onehot, prev_rew, hd])  # (83,)
```

**Key differences from Architecture 2**:
- No separate Opponent LSTM — the session LSTM IS the memory
- LSTM processes every decision point (not just between hands)
- Actor receives prev_action, prev_reward, hand_done as direct inputs
- The LSTM hidden state is the sole source of temporal context
- PPO replay is fully sequential through the LSTM (~1500 steps per rollout)

**Parameters**: Actor ~79K, Critic ~65K, Predictor ~2K, total ~146K

### Results

**Significant regression** from Architecture 2. Session evaluation: -375 mbb/h vs calling station (Architecture 2 was breakeven), -275 mbb/h vs polarizing (Architecture 2 was -140). Lost to the random player (-250 mbb/h, Architecture 2 was +550).

**Cause**: BPTT through ~1500 steps per rollout (512 hands × ~3 steps/hand) caused vanishing gradients. The LSTM couldn't learn basic poker strategy because gradients from late steps didn't reach early steps. Architecture 2's feedforward actor learned poker from clean batched gradients; Architecture 3's recurrent actor couldn't.

**Lesson**: Putting the LSTM in the actor's forward path forces it to be used but also forces all gradients through a very long recurrent chain. The cure was worse than the disease — it solved the "LSTM being ignored" problem by destroying the base policy learning.

---

## Opponent Diversity

### Hardcoded Opponents (All Architectures)

| Opponent | Strategy | Exploitable Pattern |
|----------|----------|-------------------|
| Calling Station | Always call/check | Never folds → value bet relentlessly |
| Maniac | Always raise, else call | Too aggressive → trap with strong hands |
| Old Man Coffee | Only plays AA/KK/QQ, folds everything else | Folds 94% → steal blinds constantly |
| Polarizing | Raises premiums + suited connectors, calls pairs/broadways | Predictable range → adjust accordingly |

### Parameterized Opponents (Architectures 2+)

```python
class ParameterizedPlayer:
    params: vpip (0.15-0.95), pfr (0.05-vpip), aggression (0.3-4.0), fold_to_raise (0.05-0.80)

    def act(state):
        strength = hand_strength(state.hand)  # 0-1 from pairs, high cards, suitedness
        if preflop:
            if strength < (1 - vpip): fold/check
            elif strength >= (1 - pfr): raise
            else: call
        else:
            if facing_raise and random() < fold_to_raise: fold
            if random() < aggression/(1+aggression): raise
            else: call
```

Balanced sampling across 4 categories: tight-passive, tight-aggro, loose-passive, loose-aggro (25% each).

---

## Summary Table

| | Architecture 1 | Architecture 2 | Architecture 2b | Architecture 3 |
|---|---|---|---|---|
| **Actor** | FC + Game LSTM(64) + Concat | FC + FiLM | FC + FiLM | FC + Session LSTM(64) |
| **Critic** | FC + Game LSTM(64) + Concat | Centralized (both hands) + FiLM | Centralized + FiLM | Centralized + FiLM |
| **Opponent Model** | LSTM(8→32), between hands | LSTM(8→32), between hands | LSTM(8→32), in PPO graph | Session LSTM (in actor) |
| **Conditioning** | Concatenation | FiLM (γ⊙x+β) | FiLM | Direct (LSTM output) |
| **Advantage Est.** | Monte Carlo | GAE (λ=0.95) | GAE (λ=0.95) | GAE (λ=0.95) |
| **Aux Loss** | MSE on cum. freq. | Cross-entropy on actions | Cross-entropy on actions | Cross-entropy on actions |
| **Params** | ~55K | ~107K | ~107K | ~146K |
| **Training** | PPO, 1 opponent/ep | PPO, mixed opponents | Meta-batched PPO | PPO, mixed opponents |
| **Overall Payoff** | ~0 | ~+0.4 | ~+1.0 | ~-0.5 |
| **Adaptation** | None | None | None | None (+ degraded base) |

---

## References

- Duan, Y., Schulman, J., Chen, X., Bartlett, P., Sutskever, I., & Abbeel, P. (2016). RL²: Fast Reinforcement Learning via Slow Reinforcement Learning.
- Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML.
- He, H., Boyd-Graber, J., Daumé III, H., & Eisenstein, J. (2016). Opponent Modeling in Deep Reinforcement Learning. ICML.
- Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual Reasoning with a General Conditioning Layer. AAAI.
- Rabinowitz, N., Perbet, F., Song, F., Zhang, C., Eslami, S. M. A., & Botvinick, M. (2018). Machine Theory of Mind. ICML.
- Rakelly, K., Zhou, A., Quillen, D., Finn, C., & Levine, S. (2019). Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables. ICML.
- Rudolph, M., et al. (2025). PPO with high entropy regularization outperforms specialized game-theoretic algorithms in imperfect-information games. ICLR.
- Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR.
- Srinivasan, S., et al. (2018). Actor-Critic Policy Optimization in Partially Observable Multiagent Environments. NeurIPS.
- Vinyals, O., et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature.
- Wu, Z., et al. (2022). L2E: Learning to Exploit from Expert Demonstrations in Poker. IJCAI.
- Zhao, E., et al. (2022). AlphaHoldem: High-Performance Artificial Intelligence for Heads-Up No-Limit Poker via End-to-End Reinforcement Learning. AAAI.
- Zintgraf, L., et al. (2020). VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning. ICLR.
- Bard, N., et al. (2013). Online Implicit Agent Modelling. AAMAS.
- Burch, N., et al. (2018). AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games. AAAI.
- Shi, Y., et al. (2025). AMP3: Adaptive Multi-Player Poker Playing with Policy and Style Prediction. Neural Computing and Applications.
