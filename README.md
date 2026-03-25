# Exploitative Poker Agents via Deep Reinforcement Learning

## Problem Statement

Game-theoretically optimal (GTO) strategies like CFR converge to Nash equilibrium — they cannot be exploited, but they also cannot exploit. Against suboptimal opponents, a GTO agent leaves money on the table by refusing to deviate from balanced play. This project investigates whether deep RL agents can learn to identify and exploit fixed opponent archetypes in Heads-Up Limit Texas Hold'em, outperforming CFR in expected value against those specific opponents.

## Approaches

### Baseline: CFR (Counterfactual Regret Minimization)

Nash equilibrium solver via OpenSpiel. Plays optimally in expectation against any opponent but does not adapt or exploit.

### Double DQN + LSTM

Value-based best-response agent. The LSTM encodes betting history within a hand to handle partial observability. Trained via epsilon-greedy exploration against a fixed opponent. Hypothesis: learns a near-optimal counter-strategy for each archetype.

### Actor-Critic with CFR Prior

Uses CFR's action distribution as the actor (policy prior) and a critic network (FC + LSTM) that observes game state and opponent history across hands to decide when to deviate. Hypothesis: matches CFR early, then surpasses it as the critic learns to detect exploitable patterns.

## Opponent Archetypes

| Archetype | Description |
|---|---|
| Calling Station | Passive — calls almost everything, rarely raises or folds |
| Maniac | Hyper-aggressive — raises and bets at every opportunity |
| Old Man Coffee | Tight-passive — only plays premium hands, folds the rest |

## Methodology

- **Format:** 1v1 Heads-Up Limit Hold'em (RLCard environment)
- **Training:** Each RL agent is trained against a single greedy/fixed opponent archetype
- **Evaluation:** Trained agent tested against each opponent archetype; cumulative reward compared to CFR baseline over the same number of episodes

## Repository Structure

```
agents/          # Agent implementations (CFR, RL agents)
env/             # Game environment, state/action representations
players/         # Fixed opponent archetypes
scripts/         # Training and evaluation scripts
models/          # Saved model checkpoints
results/         # Evaluation plots and logs
references/      # Papers and project proposal
```
