from dataclasses import dataclass, field


@dataclass
class Sources:
    assets: dict = field(default_factory=lambda: {})
    modified: dict = field(default_factory=lambda: {})

    def is_init(self):
        return all(self.modified.values())

    def __len__(self):
        return len(self.assets)

    def __getitem__(self, key: str):
        return self.assets[key]

    def __setitem__(self, key: str, value):
        self.modified[key] = True
        self.assets[key] = value

    def __iter__(self):
        return iter(self.assets)

    def keys(self):
        return self.assets.keys()

    def items(self):
        return self.assets.items()

    def values(self):
        return self.assets.values()


@dataclass
class Sinks:
    assets: dict = field(default_factory=lambda: {})

    def __len__(self):
        return len(self.assets)

    def __getitem__(self, key: str):
        return self.assets[key]

    def __setitem__(self, key: str, value):
        self.assets[key] = value

    def __iter__(self):
        return iter(self.assets)

    def keys(self):
        return self.assets.keys()

    def items(self):
        return self.assets.items()

    def values(self):
        return self.assets.values()
