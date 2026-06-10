# PokerBots

**Adam Ma & Ben Marler**

A research project exploring two approaches to exploiting suboptimal opponents in **Limit Texas Hold'em**: a specialist Double DQN agent and a generalist Actor-Critic agent with opponent modeling. Built on the [RLCard](https://github.com/datamllab/rlcard) environment.

**[Read the full research report](./research-report.pdf)**

---

## Overview

Modern poker solvers use Game Theory Optimal (GTO) strategies — mathematically unexploitable, but they leave money on the table against flawed human players. This project investigates whether RL agents can **dynamically exploit opponent tendencies** to do better.

Two approaches were compared:

- **The Specialist (Double DQN):** One agent trained per opponent archetype. Learns to maximally exploit a specific, fixed strategy.
- **The Generalist (Actor-Critic with Memory):** A single agent trained against a mix of all opponent types, using an LSTM to identify and adapt to opponents mid-session.

---

## Opponent Archetypes

Four rule-based opponents were implemented as training targets:

| Archetype | Behavior |
|---|---|
| **Calling Station** | Always calls, never folds |
| **Maniac** | Always raises if possible |
| **Old Man Coffee (OMC)** | Only plays premium hands (AA, KK, QQ) |
| **Polarizing** | Raises strong/nut hands, checks marginal hands, folds the rest |

> <img width="2535" height="1800" alt="image" src="https://github.com/user-attachments/assets/2eb91da9-68c9-4664-b0c8-5131f476ea70" />
> The classic 2x2 tight/loose × passive/aggressive breakdown used to categorize the four fixed opponents.

---

## State Representation

The state vector fed to all agents is a 90-dimensional binary encoding:

```
s = [c_cards, c_raises, h_rank, d_draw, c_legal, p] ∈ {0,1}^90
     52 cards + 20 raise counts + 10 hand ranks + 3 draw flags + 4 legal actions + 1 position
```

Hand rank and draw flags were explicitly added so agents don't have to learn poker hand semantics from scratch — they can focus purely on strategy.

---

## Agents

### Double DQN (Specialist)

A Dueling Double DQN with LSTM trained against a single opponent type. Key design choices:

- **Double DQN** prevents overestimation of Q-values
- **Dueling architecture** separates state value V(s) from action advantage A(s,a)
- **LSTM** captures within-hand betting history
- **Replay buffer** breaks temporal correlations for stable training
- **Huber loss** handles noisy poker rewards gracefully

Four specialist models were trained: `dqn-calling`, `dqn-maniac`, `dqn-omc`, `dqn-polar`.

> <img width="2436" height="238" alt="image" src="https://github.com/user-attachments/assets/dbb2ab67-5d97-4e9e-a749-e68fe3dd49eb" />
> A 90-dimensional state vector passed through two 128-unit ReLU hidden layers to produce Q-values for each of the 4 legal actions.

### Actor-Critic (Generalist)

Three architectures were explored, each addressing the failures of the last:

**V1 — Concatenation:** Game LSTM + Opponent LSTM, outputs concatenated. The actor learned to ignore the opponent context entirely due to high card-luck noise.

**V2 — FiLM Conditioning + CTDE:** Replaced concatenation with [Feature-wise Linear Modulation](https://arxiv.org/abs/1709.07871) (inspired by AlphaStar). The critic was given both players' hole cards during training to reduce variance. Meta-batching forced gradients to flow across hands. Reached +1.0 mbb/h overall but adaptation remained flat.

> <img width="2748" height="1320" alt="image" src="https://github.com/user-attachments/assets/b2d2398e-85d4-49a1-a13e-27155acff3b0" />
> The V2 dual-stream architecture: game state flows through an MLP trunk modulated by the opponent LSTM context via FiLM scale/shift parameters.

**V3 — Fully Recurrent RL²:** A single Session LSTM spanning all decision points in a session. Failed due to vanishing gradients across ~1500 BPTT steps.

---

## Results



> <img width="2640" height="1746" alt="image" src="https://github.com/user-attachments/assets/69e7ab46-74a6-40be-a18a-21388b88aaa6" />
> DQN-Calling dominates its target opponent (cyan), while AC models (red/orange) maintain near break-even. DQN-Maniac (dark blue) loses badly — it never learned to beat a passive player.

> <img width="2630" height="1723" alt="image" src="https://github.com/user-attachments/assets/a7e757f1-54e7-4c0a-8cba-4f4b71d69f83" />
> All DQN models profit against OMC. AC models perform well but with significantly higher variance than DQN.

> <img width="2628" height="1720" alt="image" src="https://github.com/user-attachments/assets/88bee10d-8630-49aa-a016-8576a509f3c4" />
> Surprising result: DQN-Calling crushes the Maniac (cyan), while DQN-Maniac (dark blue) barely breaks even against its own training target.

> <img width="2736" height="1795" alt="image" src="https://github.com/user-attachments/assets/801a9bfd-d7c6-4826-a0ec-540552f7df05" />
> The meta training on AC-v2 drastically stabilized the performance and it performed well with a slight upward trend

> <img width="1905" height="1755" alt="image" src="https://github.com/user-attachments/assets/66340a77-0814-4e67-adda-068c5e817d50" />
> The overlapping "First half / Second half" curves confirm that AC models are not adapting to opponents within a session: the core open problem.

**Key takeaways:**
- Specialist DQN agents strongly outperform against their training targets
- AC models are competitive but higher variance, and showed no measurable intra-session adaptation
- Meta-batched training (V2-Meta) substantially stabilized AC performance
- Surprisingly, DQN-Calling was the strongest all-around performer across multiple opponent types

---

## Project Structure

```
PokerBots/
├── agents/          # DQN and Actor-Critic agent implementations
├── env/             # RLCard environment wrappers and state encoders
├── evaluation/      # Session evaluation scripts and metrics
├── main/            # Entry point for running matchups
├── players/         # Fixed-strategy opponent implementations
├── results/         # Training curves and evaluation plots
├── train/           # Training scripts (train_dqn.py, train_ac.py, train_cfr.py)
├── references/      # Research papers and references
├── Makefile         # All commands for training, evaluation, and dev
└── pyproject.toml   # Dependencies managed via uv
```

---

## Setup

Requires **Python 3.11** and [uv](https://github.com/astral-sh/uv).

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
make sync

# Verify setup
make check-deps
```

---

## Training

```bash
# Train specialist DQN agents (one per opponent)
make train-dqn-calling
make train-dqn-maniac
make train-dqn-omc
make train-dqn-polar

# Train Actor-Critic agents
make train-ac-pure        # No KL regularization
make train-ac-kl          # With KL regularization (lambda=0.5)
```

---

## Evaluation

```bash
# Run a specific matchup (e.g., DQN-Calling vs Maniac)
make matchup AGENT=dqn-calling OPPONENT=maniac

# Run session-level evaluation for all agents vs one opponent
make sessions-all

# Or evaluate a specific pairing
make evaluate-sessions ARGS="--agents dqn-calling --opponent maniac"
```

Available agents: `ac-pure`, `ac-v2`, `ac-v2-meta`, `ac-v3`, `dqn-calling`, `dqn-maniac`, `dqn-omc`, `dqn-polar`, `random`

Available opponents: `calling`, `folder`, `maniac`, `omc`, `polar`, `random`

---

## Model Management

```bash
make pack-models      # Compress trained models to models.tar.gz
make unpack-models    # Extract models from archive
make clean-models     # Remove all .pt / .pkl weight files
```

---

## Dependencies

Key libraries (managed via `uv`):

- `rlcard` — Limit Texas Hold'em environment
- `torch 2.2.2` — Neural networks (CPU on macOS/Windows, CUDA 12.1 on Linux)
- `open-spiel` — Additional game-theoretic utilities (non-macOS)
- `numpy`, `matplotlib`, `seaborn` — Data and visualization
- `pokerkit` — Hand evaluation utilities
- `ruff` — Linting and formatting
