# Exploitative Poker Agents via Deep Reinforcement Learning

## Problem Statement

Game-theoretically optimal (GTO) strategies like CFR converge to Nash equilibrium — they cannot be exploited, but they also cannot exploit. Against suboptimal opponents, a GTO agent leaves money on the table by refusing to deviate from balanced play. Additionally, CFR is computationally expensive to both train (millions of game tree traversals) and run at inference (requires full game state reconstruction per action), making it impractical for real-time adaptive play.

This project investigates whether deep RL agents can learn to identify and exploit opponent archetypes in Heads-Up Limit Texas Hold'em, outperforming CFR in expected value while being fast at inference (a single neural network forward pass).

## Approaches

### Baseline: CFR (Counterfactual Regret Minimization)

Nash equilibrium solver via OpenSpiel. Plays optimally in expectation against any opponent but does not adapt or exploit. Computationally expensive at both training and inference time.

### Dueling Double DQN + LSTM (Adam)

Value-based best-response agent. A dueling Double DQN learns state value and action advantage while an LSTM encodes betting history within a hand to handle partial observability. Trained via epsilon-greedy exploration against a fixed opponent archetype. Hypothesis: learns a near-optimal counter-strategy for each specific archetype.

### Adaptive Actor-Critic with Opponent Modeling (Ben)

General-purpose exploitative agent that does not know which opponent it faces. Uses a dual-LSTM architecture: a game LSTM captures within-hand action sequences, and an opponent LSTM builds an implicit opponent model across hands. An adaptive KL regularization term pulls the policy toward Nash equilibrium early (when the opponent is unknown) and fades as the opponent model gains confidence, allowing the agent to exploit. Hypothesis: matches CFR early in a session, surpasses it as the opponent model identifies exploitable patterns. See [docs/ac_architecture.md](docs/ac_architecture.md) for details.

## Opponent Archetypes

| Archetype | Description |
|---|---|
| Calling Station | Passive — calls almost everything, rarely raises or folds |
| Maniac | Hyper-aggressive — raises and bets at every opportunity |
| Old Man Coffee | Tight-passive — only plays premium hands, folds the rest |

## Methodology

- **Format:** 1v1 Heads-Up Limit Hold'em (RLCard environment)
- **Training (DQN):** Trained per-archetype against a fixed opponent
- **Training (AC):** Trained against a random mixture of archetypes per episode; the opponent LSTM must identify and adapt
- **Evaluation:** Cumulative mbb/h compared to CFR baseline; early-session vs late-session performance measured to show adaptation

## Repository Structure

```
agents/          # Agent implementations (CFR, AC, RL agents)
env/             # Game environment, state/action representations
evaluation/      # Evaluation harness
players/         # Fixed opponent archetypes + agent wrappers
scripts/         # Training and evaluation scripts
models/          # Saved model checkpoints
results/         # Evaluation plots and logs
references/      # Papers and project proposal
docs/            # Architecture documentation
```

## Training and Evaluation

Train the recurrent Double DQN best-response agent against a fixed archetype:

```bash
uv run python -m train.train_dqn --name dqn_maniac --opponent maniac
```

Useful opponent names include `calling`, `maniac`, `omc`, and `polar`.

Train the adaptive actor-critic agent against the mixed-opponent curriculum:

```bash
uv run python -m train.train_ac --name ac_pure --lambda-kl 0.0
```

Evaluate one or more learned agents over many sessions:

```bash
uv run python -m evaluation.evaluate_sessions --agents ac:ac_pure dqn:dqn_maniac --opponent maniac
```

If you omit the prefix, agent names default to actor-critic, so `--agents ac_pure` still works.

For the quick matchup runner in `main.main`, the available learned agent names are
`ac-pure`, `dqn-calling`, `dqn-maniac`, `dqn-omc`, and `dqn-polar`.
