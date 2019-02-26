import torch
from copy import deepcopy

class Mean:
    """
    Running average of the values that are 'add'ed
    """
    def __init__(self, update_weight=1):
        """
        :param update_weight: 1 for normal, 2 for t-average
        """
        self.average = None
        self.counter = 0
        self.update_weight = update_weight

    def add(self, value, weight=1):
        """Add a value to the accumulator"""
        self.counter += weight
        if self.average is None:
            self.average = deepcopy(value)
        else:
            delta = value - self.average
            self.average += delta * self.update_weight * weight / (self.counter + self.update_weight - 1)
            if isinstance(self.average, torch.Tensor):
                self.average.detach()

    def value(self):
        """Access the current running average"""
        return self.average


class Max:
    """
    Keeps track of the max of all the values that are 'add'ed
    """
    def __init__(self):
        self.max = None

    def add(self, value):
        """
        Add a value to the accumulator.
        :return: `true` if the provided value became the new max
        """
        if self.max is None or value > self.max:
            self.max = deepcopy(value)
            return True
        else:
            return False

    def value(self):
        """Access the current running average"""
        return self.max
