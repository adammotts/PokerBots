from collections import Counter, defaultdict
from dataclasses import dataclass

from rlcard.games.base import Card


@dataclass
class HandStrength:
    hand_rank: int
    has_flush_draw: bool
    has_straight_draw: bool
    has_boat_draw: bool


card_ranks = {
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14,
}


def straight_window_info(cards: list[Card]) -> tuple[int, int, bool, bool]:
    """
    Returns:
        best_len: max number of ranks matched in any 5-rank straight window
        best_top: top rank of that best window
        has_straight: True if any window has all 5 ranks
        has_straight_draw: True if any window has exactly 4 of the 5 ranks

    best_top uses poker rank values:
        5 for A-2-3-4-5
        6 for 2-3-4-5-6
        ...
        14 for T-J-Q-K-A
    """
    if not cards:
        return 0, 0, False, False

    ranks = {card_ranks[card.rank] for card in cards}

    best_len = 0
    best_top = 0
    has_straight = False
    has_straight_draw = False

    for top in range(5, 14 + 1):
        if top == 5:
            window = {14, 2, 3, 4, 5}
        else:
            window = set(range(top - 4, top + 1))

        matched = len(ranks & window)

        if matched > best_len:
            best_len = matched
            best_top = top
        elif matched == best_len and top > best_top:
            best_top = top

        if matched == 5:
            has_straight = True
        elif matched == 4:
            has_straight_draw = True

    return best_len, best_top, has_straight, has_straight_draw


def evaluate_hand_strength(hand: tuple[Card, Card], board: tuple[Card]) -> HandStrength:
    """
    Evaluate hand strength
    """

    all_cards = list(hand + board)

    assert len(all_cards) <= 7

    cards_by_suit = defaultdict(list)
    for card in all_cards:
        cards_by_suit[card.suit].append(card)
    rank_freqs = Counter([card_ranks[card.rank] for card in all_cards])
    rank_freqs = sorted((freq, rank) for rank, freq in rank_freqs.items())

    hand_rank = 0

    # Royal and Straight Flush
    for cards_for_suit in cards_by_suit.values():
        best_len, best_top, has_straight, has_straight_draw = straight_window_info(
            cards_for_suit
        )

        if has_straight:
            if best_top == 14:
                if not hand_rank:
                    hand_rank = 10

            else:
                if not hand_rank:
                    hand_rank = 9

    # Quads and Boat
    if rank_freqs[-1][0] == 4:
        if not hand_rank:
            hand_rank = 8

    if rank_freqs[-1][0] == 3 and len(rank_freqs) >= 2 and rank_freqs[-2][0] >= 2:
        if not hand_rank:
            hand_rank = 7

    # Flush
    if any(len(cards_for_suit) >= 5 for cards_for_suit in cards_by_suit.values()):
        if not hand_rank:
            hand_rank = 6

    # Straight
    best_len, best_top, has_straight, has_straight_draw = straight_window_info(
        all_cards
    )
    if has_straight:
        if not hand_rank:
            hand_rank = 5

    # Set
    if rank_freqs[-1][0] == 3 and len(rank_freqs) >= 2 and rank_freqs[-2][0] < 2:
        if not hand_rank:
            hand_rank = 4

    # Two Pair
    if rank_freqs[-1][0] == 2 and len(rank_freqs) >= 2 and rank_freqs[-2][0] == 2:
        if not hand_rank:
            hand_rank = 3

    # Pair
    if rank_freqs[-1][0] == 2:
        if not hand_rank:
            hand_rank = 2

    if not hand_rank:
        hand_rank = 1

    has_flush_draw = len(all_cards) < 7 and any(
        len(cards_for_suit) == 4 for cards_for_suit in cards_by_suit.values()
    )
    has_straight_draw = len(all_cards) < 7 and not has_straight and has_straight_draw
    has_boat_draw = len(all_cards) < 7 and (hand_rank == 3 or hand_rank == 4)

    return HandStrength(
        hand_rank=hand_rank,
        has_flush_draw=has_flush_draw,
        has_straight_draw=has_straight_draw,
        has_boat_draw=has_boat_draw,
    )
