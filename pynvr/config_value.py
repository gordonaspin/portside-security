class ConfigValue:
    def __init__(self, default, min, max, step):
        self.default = default
        self.min = min
        self.value = default
        self.max = max
        self.step = step