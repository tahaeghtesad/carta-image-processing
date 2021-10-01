class FixedSizeBuffer:
    def __init__(self, max_size) -> None:
        super().__init__()
        self.max_size = max_size
        self.buffer = []

    def add(self, item):
        self.buffer.append(item)
        if len(self.buffer) > self.max_size:
            del self.buffer[0]
