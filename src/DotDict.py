class DotDict:
    """Dictionary subclass that allows dot notation access to keys, used by the fake streamlit proxy."""
    def __init__(self, dictionary=None):
        self._dict = {} if dictionary is None else dictionary

    def __getattr__(self, key):
        # If the value is a dict, return it as a DotDict
        value = self._dict.get(key)
        if isinstance(value, dict):
            return value
        return value

    def __setattr__(self, key, value):
        if key == '_dict':
            super().__setattr__(key, value)
        else:
            self._dict[key] = value

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        self._dict[key] = value

    def get(self, key, default=None):
        return self._dict.get(key, default)

    def clear(self):
        self._dict.clear()

    def copy(self):
        return DotDict(self._dict.copy())

    def update(self, other):
        self._dict = other
