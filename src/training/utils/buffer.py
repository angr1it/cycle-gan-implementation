import random

import torch
from torch import Tensor
from torch.autograd import Variable


class ReplayBuffer:
    def __init__(self, size=100) -> None:
        self.buffer: list[Tensor] = []
        self.size = size

    def process(self, sample):
        res = []

        for el in sample:
            el = torch.unsqueeze(el, 0)

            if len(self.buffer) < self.size:
                self.buffer.append(el)
                res.append(el)
                continue

            elif random.uniform(0, 1) > 0.5:
                res.append(el)
                continue

            i = random.randint(0, self.size - 1)
            res.append(self.buffer[i].clone())
            self.buffer[i] = el

        return Variable(torch.cat(res))
