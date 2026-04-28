# nice ideas from github/danijar/embodied


class Every:
    """Executes every N steps.

    ```python
    should_print = Every(n=100)
    for _ in range(1_000):
        if should_print():
            print("Test")
    # >>> executes 10 times
    ```
    """

    def __init__(self, n: int) -> None:
        self._n = n
        self._count = 0

    def __call__(self) -> bool:
        self._count = (self._count + 1) % self._n
        return self._count == 0


class Ratio:
    """Generalization of `Every` that supports multiple executions in one go."""

    def __init__(self, ratio: float) -> None:
        self._ratio = ratio
        self._budget = 0.0

    def __call__(self) -> int:
        self._budget += self._ratio
        steps = int(self._budget)
        self._budget -= steps
        return steps
