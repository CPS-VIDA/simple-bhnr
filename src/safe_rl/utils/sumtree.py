"""Sum Tree implementation"""
import numpy as np


class SumTree:
    """
    Sum Tree Implementation. The root of any subtree is the sum of all its sub trees.
    """

    def __init__(self, capacity):
        self._capacity = capacity
        self._tree = np.zeros(2 * capacity - 1)
        self._data = np.zeros(capacity, dtype=object)
        self._size = 0
        self._pos = 0

    def update(self, idx, p):
        idx = int(idx)
        delta = p - self._tree[idx]
        self._tree[idx] += delta
        while idx != 0:
            parent = (idx - 1) // 2
            self._tree[parent] += delta
            idx = parent

    @property
    def capacity(self):
        return self._capacity

    def __len__(self):
        return self._size

    @property
    def total(self):
        return self._tree[0]

    def add(self, data, p):
        idx = self._pos + self.capacity - 1
        self._data[self._pos] = data
        self.update(idx, p)
        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def _retreive(self, idx, num):
        left = 2 * idx + 1
        right = left + 1

        while left < len(self._tree):
            if num <= self._tree[left]:
                idx = left
            else:
                idx = right
                num = num - self._tree[left]

            left = 2 * idx + 1
            right = left + 1
        return idx

    def sample(self, num):
        idx = self._retreive(0, num)
        data_idx = idx - self.capacity + 1
        return idx, self._tree[idx], self._data[data_idx]

    def __iter__(self):
        return iter(zip(self._data, self._tree[self.capacity - 1]))
