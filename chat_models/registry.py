from __future__ import annotations

from collections.abc import Callable

from chat_models.base import ChatStrategy


class StrategyRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], ChatStrategy]] = {}

    def register(self, name: str, factory: Callable[[], ChatStrategy]) -> None:
        if name in self._factories:
            raise ValueError(f"strategy already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str) -> ChatStrategy:
        try:
            return self._factories[name]()
        except KeyError as exc:
            choices = ", ".join(sorted(self._factories))
            raise ValueError(f"unknown MODEL_STRATEGY '{name}'; choose one of: {choices}") from exc

    def names(self) -> list[str]:
        return sorted(self._factories)
