import torch


class SimpleBatcher:
    def __init__(self, indices: torch.Tensor, batch_size: int, shuffle: bool = True):
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.indices = self.indices[torch.randperm(len(self.indices))]
        for i in range(0, len(self.indices), self.batch_size):
            yield self.indices[i : i + self.batch_size]
