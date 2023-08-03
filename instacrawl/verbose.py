"""This module contains functions for diagnosing the state of the program."""

import threading
import traceback
import sys


def get_stacktrace() -> str:
    """Return a string containing the stacktrace of all threads."""
    stacktrace = ""
    for th in threading.enumerate():
        stacktrace += str(th)
        current_thread = th.ident
        if current_thread is not None:
            stacktrace += "".join(
                traceback.format_stack(
                    sys._current_frames()[current_thread]
                )
            )
    return stacktrace
