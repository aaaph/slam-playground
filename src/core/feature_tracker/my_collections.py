import copy


class ResettableDict(dict):
    """Dictionary that resets to default values on clear()."""

    def __init__(self, defaults: dict) -> None:
        """Initialize the dictionary."""
        super().__init__(copy.deepcopy(defaults))
        self._default = copy.deepcopy(defaults)

    def clear(self) -> None:
        """Clear the dictionary and reset to default values."""
        super().clear()
        super().update(copy.deepcopy(self._default))
