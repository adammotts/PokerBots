import os
import pickle

import numpy as np
import pyspiel
from open_spiel.python.algorithms import external_sampling_mccfr as es_mccfr

from agents.base_agent import BaseAgent, Transition
from env.state import State

GAME_PARAMS = {
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

RANK_TO_INDEX = {
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
SUIT_TO_INDEX = {"C": 0, "D": 1, "H": 2, "S": 3}

RLCARD_ACTION_TO_OS = {"fold": 0, "call": 1, "check": 1, "raise": 2}

OS_TO_RLCARD = {0: [2], 1: [0, 3], 2: [1]}


def rlcard_card_to_os_action(card: str) -> int:
    """Convert RLCard card (e.g. 'HQ', 'DA') to OpenSpiel deal action ID."""
    suit_index = SUIT_TO_INDEX[card[0]]
    rank_index = RANK_TO_INDEX[card[1:]]
    return rank_index * 4 + suit_index


class CFRAgent(BaseAgent):
    """CFR agent using OpenSpiel's External Sampling MCCFR (C++ backend).

    Trains ~10,000x faster than RLCard's pure-Python CFR on limit holdem.
    Implements BaseAgent so it's interchangeable with the RLCard CFR wrapper.
    """

    def __init__(self, iterations: int = 1000) -> None:
        self.iterations = iterations
        self.total_iterations: int = 0
        self.game = pyspiel.load_game("universal_poker", GAME_PARAMS)
        self.solver = es_mccfr.ExternalSamplingSolver(self.game)
        self.avg_policy: es_mccfr.AveragePolicy | None = None

    def act(
        self,
        *,
        state: State,
        training: bool = True,
        action_record: list[tuple[int, str]] | None = None,
    ) -> int:
        if state.raw_obs is None:
            return int(np.random.choice(state.legal_actions))

        if self.avg_policy is None:
            self.avg_policy = self.solver.average_policy()

        os_state = self.build_info_state(
            state.raw_obs, action_record or [], state.player_id
        )
        try:
            probs = self.avg_policy.action_probabilities(os_state)
        except (IndexError, KeyError, pyspiel.SpielError):
            return int(np.random.choice(state.legal_actions))

        legal_set = set(state.legal_actions.keys())
        prob_array = np.zeros(4)
        for os_action, prob in probs.items():
            for rl_action in OS_TO_RLCARD[os_action]:
                if rl_action in legal_set:
                    prob_array[rl_action] += prob

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
            self.solver.iteration()
            self.total_iterations += 1
        self.avg_policy = None

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        final = os.path.join(path, "cfr.pkl")
        tmp = final + ".tmp"
        with open(tmp, "wb") as f:
            p = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
            p.fast = True
            p.dump(self.total_iterations)
            for key, val in self.solver._infostates.items():
                p.dump((key, val))
            p.dump(None)
        os.replace(tmp, final)

    def load(self, path: str) -> None:
        pkl = os.path.join(path, "cfr.pkl")
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
            self.solver._infostates = infostates
        except (EOFError, pickle.UnpicklingError) as e:
            print(
                f"[OpenSpielCFRAgent] Warning: checkpoint corrupt ({e}), starting fresh"
            )
            return
        self.avg_policy = None

    def build_info_state(
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
        state = self.game.new_initial_state()

        our_cards = [rlcard_card_to_os_action(c) for c in raw_obs["hand"]]
        public_cards = [rlcard_card_to_os_action(c) for c in raw_obs["public_cards"]]
        used = set(our_cards + public_cards)

        dummy_opp = [i for i in range(52) if i not in used][:2]

        if player_id == 0:
            deal_order = our_cards + dummy_opp
        else:
            deal_order = dummy_opp + our_cards

        for card_action in deal_order:
            state.apply_action(card_action)

        public_index = 0
        action_index = 0
        while not state.is_terminal():
            if state.is_chance_node():
                if public_index < len(public_cards):
                    state.apply_action(public_cards[public_index])
                    public_index += 1
                else:
                    break
            elif state.current_player() < 0:
                break
            elif action_index < len(action_record):
                _, action_str = action_record[action_index]
                action_index += 1
                os_action = RLCARD_ACTION_TO_OS.get(action_str, 1)
                if os_action in state.legal_actions():
                    state.apply_action(os_action)
                else:
                    state.apply_action(1)
            else:
                break

        return state
