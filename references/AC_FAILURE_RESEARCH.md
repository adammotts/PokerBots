# Refactoring a poker bot that can't exploit its opponents

**Your agent's core failure is not architectural — it's a training design problem.** The opponent LSTM receives no useful gradient because poker's card-luck variance drowns the adaptation signal, and your training loop doesn't create the right learning pressure for within-session adaptation. The fix requires three coordinated changes: a **centralized critic** that eliminates card-luck variance during training, **supervised auxiliary losses** that give the opponent model clean gradient signal independent of reward noise, and an **RL²-style meta-training loop** that explicitly rewards adaptation speed rather than static performance. These changes should produce measurable exploitation gains against diverse opponents within a single-GPU training budget of 2–5 million hands (4–12 hours).

---

## Priority 1: A centralized critic eliminates your variance problem

The single highest-impact change is training a **centralized critic that observes both players' hole cards** during training, while the actor sees only its own. This is the Centralized Training, Decentralized Execution (CTDE) paradigm from multi-agent RL (Srinivasan et al., NeurIPS 2018). Since the ~100× variance ratio you identified comes almost entirely from card luck — the agent doesn't know if it lost because it played badly or because the opponent held aces — a critic that sees both hands can perfectly attribute outcomes to decisions versus luck. The actor remains information-constrained and plays legally; only the critic gets privileged access.

**GAE replaces your Monte Carlo returns.** Switch from pure MC returns to Generalized Advantage Estimation with **λ = 0.90–0.95** and **γ = 1.0** (appropriate for finite-horizon poker hands). Each betting round (preflop, flop, turn, river) becomes a natural timestep for GAE computation. With a well-trained centralized critic, GAE bootstraps from the critic's card-informed value estimates at each round boundary, providing dramatically lower-variance advantage estimates. The original GAE paper (Schulman et al., ICLR 2016) recommends λ = 0.95–0.97 for continuous control, but poker's extreme variance warrants **more aggressive bias introduction via lower λ**.

**AIVAT-inspired corrections provide further reduction.** AIVAT (Burch et al., AAAI 2018) — the variance reduction technique behind DeepStack's evaluation — achieves **85% standard deviation reduction** in heads-up no-limit by subtracting control variates for each chance event. While AIVAT was designed for evaluation rather than training, you can adapt its core idea: after each hand, compute a correction term `E[V(random_card)] - V(actual_card)` using your critic network and subtract it from the return. This removes the component of reward attributable to being dealt good or bad cards. You know your own policy (required for AIVAT's action corrections) but not the opponent's, so you can only correct for card luck and your own action variance — still a substantial reduction.

**Three additional changes compound the effect.** First, increase entropy regularization to **0.05–0.2**, far above PPO's typical 0.01 default. Rudolph et al. (ICLR 2025) demonstrated across 7,000+ training runs that PPO with high entropy regularization consistently outperformed specialized game-theoretic algorithms (NFSP, PSRO, ESCHER, R-NaD) in imperfect-information games — the community had dismissed PPO for these settings due to insufficient hyperparameter tuning. Second, perform **5–10 critic update steps per actor update** (Srinivasan et al., 2018 found this ratio critical for poker). Third, increase your batch size substantially — **512–1024 hands per PPO update** rather than 128 — to further average out residual variance.

---

## Priority 2: The opponent model needs its own gradient signal

Your diagnosed joint-learning problem — the actor needs useful opponent context to learn exploitation, but the opponent LSTM needs useful actor gradients to learn what context to provide — is well-documented in the opponent modeling literature. The solution is **decoupling the opponent model's training from the RL reward signal** by adding supervised auxiliary losses.

**Add a supervised action-prediction head.** Attach a small MLP to the opponent LSTM's hidden state that predicts the opponent's next action (fold/call/raise/check) at each decision point. Train this head with cross-entropy loss weighted at **0.3–0.5× the PPO loss**. This provides clean, per-decision gradient signal that forces the opponent LSTM to actually encode behavioral patterns. Both DRON (He et al., ICML 2016) and ToMnet (Rabinowitz et al., ICML 2018) demonstrate that supervised prediction objectives produce far better opponent representations than end-to-end RL alone. Papoudakis and Albrecht (2021) explicitly reported they "did not manage to optimize the [opponent] representation jointly with the policy" using A2C — the gradient was too noisy. A second auxiliary head predicting **opponent hand strength category at showdown** (when available) adds another clean signal source.

**Switch from concatenation to FiLM conditioning.** Your current design concatenates the 32-dim opponent context to the 77-dim game state, requiring the first hidden layer to implicitly learn multiplicative interactions. Feature-wise Linear Modulation (FiLM; Perez et al., AAAI 2018) instead uses the opponent embedding to generate per-layer **scale (γ) and shift (β) parameters** that modulate the policy network's hidden activations: `output = γ ⊙ hidden + β`. This allows the opponent representation to gate and amplify specific feature channels at every layer. **AlphaStar used FiLM conditioning** in its architecture (Vinyals et al., Nature 2019), and it has been validated in multi-agent RL. Implementation is trivial — one extra linear layer per FiLM-conditioned layer producing `(γ, β)` from the opponent embedding.

**Enrich the opponent representation with explicit statistics.** Your 8-dim hand summaries are too compressed. Maintain running tallies of interpretable poker statistics computed directly (no learning required): **VPIP** (voluntarily put money in pot), **PFR** (preflop raise frequency), **aggression factor** (raises÷calls), **fold-to-bet frequency** per street, and **went-to-showdown percentage**. Feed these alongside the raw action sequences into the opponent LSTM. The explicit stats provide a reliable baseline signal after just 20–30 hands, which the LSTM can augment with sequential pattern detection. AMP3 (Shi et al., Neural Computing and Applications, 2025) demonstrated this hybrid approach — explicit style features predicted from history and fed into an Actor-Critic policy — enables effective adaptation in multi-player Texas Hold'em.

**Consider Mixture-of-Experts as an alternative to FiLM.** DRON-MoE (He et al., ICML 2016) maintains K separate policy heads (one per hypothesized opponent archetype) and uses a gating network conditioned on opponent features to blend them: `π(a|s) = Σ_k gate_k · π_k(a|s)`. With **K = 4–6 experts**, this provides a strong structural prior for poker where there are arguably a small number of meaningful response strategies (tight-aggressive counter, loose-passive exploit, etc.). DRON-MoE consistently outperformed DRON-Concat across evaluation domains. If FiLM alone doesn't produce visible adaptation, MoE is the next experiment to try.

---

## Priority 3: Restructure training as meta-learning with RL²

Your current architecture is already 90% of the way to RL² (Duan et al., 2016) — an LSTM policy trained with PPO where the hidden state persists across episodes. **The critical missing ingredient is the meta-training loop structure.** In your current setup, the agent is optimized to maximize reward against whichever single opponent it faces, and the opponent LSTM is just an auxiliary module. In RL², the entire system is optimized to maximize **cumulative reward across an entire trial** (session of N hands against one opponent), which explicitly rewards learning speed — an agent that identifies and exploits the opponent faster earns more cumulative reward.

**RL² is the right meta-RL framework here, not PEARL or MAML.** PEARL (Rakelly et al., ICML 2019) requires SAC and cannot straightforwardly work with PPO — the authors explicitly state they could not optimize the context encoder with on-policy methods. MAML (Finn et al., ICML 2017) requires inner-loop gradient updates during the session, which is problematic in poker: each gradient step needs ~50+ hands for a stable estimate in a high-variance game, leaving almost no hands for exploitation. RL² adapts at every timestep through hidden-state forward passes with **zero test-time compute overhead**. The pytorch_rl2 codebase (github.com/lucaslingle/pytorch_rl2) confirms single-GPU feasibility with PPO.

**The concrete RL² poker training loop looks like this:**

```
for meta_iteration in range(num_meta_iterations):
    trial_batch = []
    for trial in range(meta_batch_size):          # e.g., 16-32 trials
        opponent = sample_opponent(distribution)    # from diverse pool
        hidden_state = zeros()                      # RESET only here
        trial_trajectory = []
        
        for hand in range(hands_per_trial):         # e.g., 100 hands
            for decision_point in hand:
                input = [obs, prev_action, prev_reward, hand_done_flag]
                action, value, hidden_state = rnn_policy(input, hidden_state)
                # hidden_state PERSISTS across hands — this IS the adaptation
            trial_trajectory.append(hand_data)
        trial_batch.append(trial_trajectory)
    
    ppo_update(rnn_policy, trial_batch)  # optimize for cumulative trial return
```

**The key structural differences from your current loop are:** (1) the hidden state persists across all hands within a trial and resets only when sampling a new opponent, (2) the PPO objective optimizes for cumulative return across the entire trial (not per-hand), and (3) each meta-batch contains trials against many different opponents. The `hand_done_flag` is fed as an input token so the RNN knows hand boundaries, but the hidden state flows through continuously.

**Your 4 opponents are grossly insufficient.** L2E (Wu et al., IJCAI 2022) — the most directly relevant paper, which applied MAML-style meta-learning to poker exploitation in Leduc Hold'em — demonstrated that **automatically generated diverse opponents** are critical. With only 4 deterministic types, the meta-learner memorizes a lookup table rather than learning generalizable adaptation. Create a `ParameterizedOpponent` class parameterized by continuous values for VPIP (15–85%), PFR (5–50%), aggression factor (0.3–5.0), and fold-to-raise frequency (10–80%). Sample these parameters from a uniform or Dirichlet distribution each trial. Target **30–50+ distinct strategies** in the training distribution, including a near-Nash opponent so the agent maintains a safety floor. L2E's Diverse-OSG module (which generates diverse opponents via MMD-regularized optimization) significantly outperformed both hand-coded archetypes and purely adversarial opponent generation.

**VariBAD is a worthwhile stretch goal** if time permits. VariBAD (Zintgraf et al., ICLR 2020) adds a VAE encoder that maintains an explicit **probabilistic belief distribution over opponent types**, trained jointly with PPO. This gives you interpretable uncertainty — you can visualize the agent's posterior belief shifting from "unsure" to "this is a calling station" over the first 30 hands. VariBAD's code (github.com/lmzintgraf/varibad) uses PPO and is well-documented.

---

## Priority 4: Practical changes to make all this work

**Remove the within-hand game LSTM.** RLCard's 72-dim observation already encodes raise counts per round as one-hot vectors (indices 52–71), which captures the essential within-hand betting history. Every major poker RL system — NFSP, DREAM, DeepStack, AlphaHoldem — uses **feedforward networks** for within-hand decision-making when the observation encodes sufficient history. Your game LSTM adds complexity (BPTT, hidden state management, truncation decisions) for marginal information gain (temporal ordering of actions within a round, which has near-zero strategic value in limit hold'em with max 4 raises per round). Replace it with an MLP. Keep recurrence only for the cross-hand opponent model where temporal memory genuinely matters.

**Pretrain from CFR via behavioral cloning.** Starting from a near-equilibrium policy provides three benefits: faster convergence, a safety floor against competent opponents, and a reduced learning problem (learning *when to deviate* from Nash rather than learning poker from scratch). Several open-source CFR implementations exist for heads-up limit hold'em:

- **open-pure-cfr** (github.com/rggibson/open-pure-cfr): Pure CFR for ACPC games including limit hold'em
- **slumbot2019** (github.com/ericgjackson/slumbot2019): Mature CFR+/MCCFR implementation
- **RLCard's built-in CFR agent**: Works with `allow_step_back=True`, though slow for full limit hold'em without card abstraction
- **PokerRL** (github.com/EricSteinberger/PokerRL): Vanilla CFR, CFR+, and Linear CFR implementations

Generate ~100K hands of CFR self-play data and train your policy network to reproduce the **full action probability distributions** (not argmax actions) using KL divergence loss for ~50–100 epochs. This preserves the mixed-strategy nature of poker. Then switch to the RL² meta-training loop, where the pretrained policy serves as the starting point that learns to deviate for exploitation.

**Realistic training budget on a single GPU:**

| Phase | Hands | Estimated Time |
|---|---|---|
| CFR data generation | 100K | 30–60 min |
| Behavioral cloning | — | 10–20 min |
| RL² meta-training | 2–5M | 4–12 hours |
| Evaluation (per opponent) | 50–100K | minutes |

Your ~55K parameter budget is appropriate for limit hold'em. DREAM uses 64-unit networks (~similar scale), NFSP uses [128, 128] layers (~20K params), and limit hold'em's constrained action space means overfitting is a bigger risk than underfitting. AlphaHoldem's success with PPO in 3 days on one PC (Zhao et al., AAAI 2022) validates the approach at larger scale.

---

## Revised architecture and training plan

The recommended architecture replaces your dual-LSTM + concatenation design with a streamlined system:

```
OBSERVATIONS (72-dim from RLCard)
    │
    ▼
Base Encoder: MLP(72 → 128 → 128)  ──────────────── Base features (128-dim)
                                                           │
OPPONENT ACTION HISTORY (per-hand summaries)                │
    │                                                       │
    ▼                                                       │
Explicit Stats: VPIP, PFR, AF, fold-freq (8-dim)           │
    │                                                       │
    ├──► Opponent GRU(16-dim input → 64-dim hidden)         │
    │         │                                             │
    │    Aux Head: predict next opponent action (supervised) │
    │         │                                             │
    │    Opponent embedding (64-dim) ──► FiLM Generator     │
    │                                    │     │            │
    │                                   (γ₁,β₁) (γ₂,β₂)   │
    │                                    │     │            │
    │                           ┌────────┘     │            │
    │                           ▼              ▼            │
    │                    FiLM Layer 1    FiLM Layer 2       │
    │                           │              │            │
    ▼                           ▼              ▼            │
Centralized Critic ◄───── Policy Head: → action logits     │
(sees both hands)         MLP(128→64→4)                    │
    │                                                       │
    ▼                                                       
Value estimate (used only in training)
```

**Total parameters: ~45–60K.** The centralized critic adds ~15K params but is used only during training. At deployment, only the actor + opponent GRU + FiLM generator execute (~35K params, fast inference).

**The training plan proceeds in three phases:**

**Phase 1 (days 1–3): Baseline and pretraining.** Generate CFR data and behavioral-clone a feedforward policy. Implement the centralized critic and verify it reduces per-hand return variance by measuring standard deviation of advantages before and after. Implement 20+ parameterized opponents. Target: pretrained policy beats random agent by >200 mbb/h.

**Phase 2 (days 4–8): RL² meta-training.** Implement the full meta-training loop with opponent GRU, FiLM conditioning, and supervised auxiliary loss. Train with 100-hand trials, meta-batch size 16–32, PPO with 3–4 epochs, entropy coefficient 0.1, λ = 0.93. Run for 2–5M hands total. Monitor the **first-half vs. second-half trial payoff gap** — this is your primary adaptation metric. Target: second-half payoff exceeds first-half by >50 mbb/h.

**Phase 3 (days 9–12): Evaluation and ablation.** Test against held-out opponent types not seen in training. Run 1000-hand sessions and plot cumulative reward curves — a successfully adapting agent shows an inflection point around hands 30–80 where exploitation kicks in. Compare against your CFR-pretrained policy (no adaptation) as the baseline. Run ablations removing each component (centralized critic, FiLM, auxiliary loss, opponent diversity) to verify each contributes.

---

## What to realistically expect

Against exploitable opponents (calling stations, maniacs), a well-trained RL²-style agent should achieve **100–500 mbb/h exploitation** after adaptation, compared to ~0–50 mbb/h for a static Nash-approximate strategy. Against near-Nash opponents, the agent should fall back to approximately break-even play (the CFR pretraining provides this safety floor). The adaptation should be visible in cumulative reward plots within **50–100 hands**.

The most likely failure mode is insufficient opponent diversity during meta-training — if the agent encounters an opponent type far outside its training distribution, adaptation will be slow or fail. The mitigation is L2E-style diverse opponent generation with continuous parameterization covering the strategy space broadly. A secondary failure mode is the centralized critic overfitting to specific card combinations rather than learning generalizable value estimates; regularization and large batch sizes address this.

The Bayesian portfolio approach (Bard et al., AAMAS 2013) — which pre-computes counter-strategies against K prototype opponents offline, then uses the Exp4 bandit algorithm to select among them during play — **won the ACPC 2011 opponent exploitation event** and requires no gradient-based opponent modeling at all. If the neural approach proves too noisy despite all variance reduction, this is a strong fallback that sidesteps the gradient problem entirely.

## Conclusion

The refactored system's key insight is separating three learning problems that your current architecture conflates. The **base policy** learns poker fundamentals via behavioral cloning from CFR. The **opponent model** learns to recognize behavioral patterns via supervised action prediction. The **meta-learner** (RL² training loop) learns when and how to deviate from equilibrium for exploitation. Each component receives clean gradient signal through its own loss function rather than all three fighting over a single noisy reward. The centralized critic removes card-luck variance from the training signal, making the remaining adaptation signal learnable. Combined with adequate opponent diversity (~30–50 parameterized types), this architecture should demonstrate clear within-session exploitation on a single GPU within a week of development and training.