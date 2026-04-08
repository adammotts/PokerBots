from enum import Enum


class Action(Enum):
    CALL = 0
    RAISE = 1
    FOLD = 2
    CHECK = 3


ACTION_NAMES = {a.value: a.name.lower() for a in Action}
