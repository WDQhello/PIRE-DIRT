from __future__ import annotations

from .distributed import is_main_process


def rank_zero_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)
