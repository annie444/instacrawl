"""The stoage objects that are used in the program."""

from enum import Enum


class Steps(str, Enum):
    """The steps that have been completed."""

    none = "0"
    one = "1"
    two = "2"
