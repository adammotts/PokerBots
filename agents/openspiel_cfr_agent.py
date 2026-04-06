import os
import pickle

import numpy as np
import numpy.typing as npt
import pyspiel
from env.state import State
from open_spiel.python.algorithms import external_sampling_mccfr as es_mccfr

from agents.base_agent import BaseAgent, Transition

# Limit holdem game config for OpenSpiel
_GAME_PARAMS = {
    "betting": "limit",
    "numPlayers": 2,
    "numRounds": 4,
    "blind": "10 5",
    "raiseSize": "10 10 20 20",
    "firstPlayer": "2 1 1 1",
    "maxRaises": "4 4 4 4",
    "numSuits": 4,
    "numRanks": 13,
    "numHoleCards": 2,
    "numBoardCards": "0 3 1 1",
}

# OpenSpiel card encoding: action = rank_index * 4 + suit_index
_RANK_TO_IDX = {
    "2": 0,
    "3": 1,
    "4": 2,
    "5": 3,
    "6": 4,
    "7": 5,
    "8": 6,
    "9": 7,
    "T": 8,
    "J": 9,
    "Q": 10,
    "K": 11,
    "A": 12,
}
_SUIT_TO_IDX = {"C": 0, "D": 1, "H": 2, "S": 3}

# OpenSpiel action IDs: 0=fold, 1=call/check, 2=raise/bet
_RLCARD_ACTION_TO_OS = {"fold": 0, "call": 1, "check": 1, "raise": 2}

# RLCard action IDs: 0=call, 1=raise, 2=fold, 3=check
# Map OpenSpiel action → RLCard action(s)
_OS_TO_RLCARD = {0: [2], 1: [0, 3], 2: [1]}  # OS call/check maps to both


def _rlcard_card_to_os_action(card: str) -> int:
    """Convert RLCard card (e.g. 'HQ', 'DA') to OpenSpiel deal action ID."""
    suit_idx = _SUIT_TO_IDX[card[0]]
    rank_idx = _RANK_TO_IDX[card[1:]]
    return rank_idx * 4 + suit_idx


class OpenSpielCFRAgent(BaseAgent):
    """CFR agent using OpenSpiel's External Sampling MCCFR (C++ backend).

    Trains ~10,000x faster than RLCard's pure-Python CFR on limit holdem.
    Implements BaseAgent so it's interchangeable with the RLCard CFR wrapper.
    """

    def __init__(self, iterations: int = 1000) -> None:
        self.iterations = iterations
        self.total_iterations: int = 0
        self._game = pyspiel.load_game("universal_poker", _GAME_PARAMS)
        self._solver = es_mccfr.ExternalSamplingSolver(self._game)
        self._avg_policy: es_mccfr.AveragePolicy | None = None

    def act(
        self,
        *,
        state: State,
        training: bool = True,
        action_record: list[tuple[int, str]] | None = None,
    ) -> int:
        if state.raw_obs is None:
            return int(np.random.choice(state.legal_actions))

        if self._avg_policy is None:
            self._avg_policy = self._solver.average_policy()

        os_state = self._build_info_state(state.raw_obs, action_record or [], state.player_id)
        try:
            probs = self._avg_policy.action_probabilities(os_state)
        except (IndexError, KeyError, pyspiel.SpielError):
            return int(np.random.choice(state.legal_actions))

        # Map OpenSpiel action probs → RLCard 4-action space
        legal_set = set(state.legal_actions.keys())
        prob_array = np.zeros(4)
        for os_action, prob in probs.items():
            for rl_action in _OS_TO_RLCARD[os_action]:
                if rl_action in legal_set:
                    prob_array[rl_action] += prob

        # Zero out illegal actions and renormalize
        mask = np.zeros(4)
        for a in state.legal_actions:
            mask[a] = 1.0
        prob_array *= mask
        total = prob_array.sum()
        if total > 0:
            prob_array /= total
        else:
            prob_array[list(state.legal_actions)] = 1.0 / len(state.legal_actions)

        return int(np.random.choice(4, p=prob_array))

    def observe(self, transition: Transition) -> None:
        pass

    def update(self) -> None:
        for _ in range(self.iterations):
            self._solver.iteration()
            self.total_iterations += 1
        self._avg_policy = None  # invalidate cached policy

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        final = os.path.join(path, "openspiel_cfr.pkl")
        tmp = final + ".tmp"
        with open(tmp, "wb") as f:
            p = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
            # fast=True disables pickle's memo dict (cycle detection/deduplication).
            # For millions of infostates the memo alone can consume several GB of RAM,
            # causing OOM on save. Infostates have no shared references so this is safe.
            p.fast = True
            p.dump(self.total_iterations)
            for key, val in self._solver._infostates.items():
                p.dump((key, val))
            p.dump(None)  # sentinel marks end of stream
        os.replace(tmp, final)  # atomic on Linux (rename syscall)

    def load(self, path: str) -> None:
        pkl = os.path.join(path, "openspiel_cfr.pkl")
        if not os.path.exists(pkl):
            return
        try:
            with open(pkl, "rb") as f:
                u = pickle.Unpickler(f)  # noqa: S301
                self.total_iterations = u.load()
                infostates = {}
                while True:
                    item = u.load()
                    if item is None:
                        break
                    key, val = item
                    infostates[key] = val
            self._solver._infostates = infostates
        except (EOFError, pickle.UnpicklingError) as e:
            print(
                f"[OpenSpielCFRAgent] Warning: checkpoint corrupt ({e}), starting fresh"
            )
            return
        self._avg_policy = None

    def _build_info_state(
        self,
        raw_obs: dict[str, object],
        action_record: list[tuple[int, str]],
        player_id: int = 0,
    ) -> pyspiel.State:
        """Reconstruct an OpenSpiel State from RLCard state.

        Replays the exact card deals and action history on a fresh OpenSpiel
        state. Deal order: P0-card1, P0-card2, P1-card1, P1-card2, then
        community cards between betting rounds. Opponent's cards don't
        affect our info state, so we deal arbitrary unused cards for them.
        """
        state = self._game.new_initial_state()

        our_cards = [_rlcard_card_to_os_action(c) for c in raw_obs["hand"]]
        public_cards = [_rlcard_card_to_os_action(c) for c in raw_obs["public_cards"]]
        used = set(our_cards + public_cards)

        # Pick 2 dummy cards for the opponent (any unused cards)
        dummy_opp = [i for i in range(52) if i not in used][:2]

        # Deal order: P0-c1, P0-c2, P1-c1, P1-c2
        if player_id == 0:
            deal_order = our_cards + dummy_opp
        else:
            deal_order = dummy_opp + our_cards

        for card_action in deal_order:
            state.apply_action(card_action)

        # Replay action history, dealing community cards between rounds
        public_idx = 0
        action_idx = 0
        while not state.is_terminal():
            if state.is_chance_node():
                if public_idx < len(public_cards):
                    state.apply_action(public_cards[public_idx])
                    public_idx += 1
                else:
                    break
            elif state.current_player() < 0:
                # Terminal or chance — stop
                break
            elif action_idx < len(action_record):
                _, action_str = action_record[action_idx]
                action_idx += 1
                os_action = _RLCARD_ACTION_TO_OS.get(action_str, 1)
                if os_action in state.legal_actions():
                    state.apply_action(os_action)
                else:
                    state.apply_action(1)
            else:
                break

        return state
