import numpy as np

class Hole():
    def __init__(self, x, y):
        self.pos = np.array([x, y])

class TableSide():
    def __init__(self, line):
        self.line = np.array(line)
        self.middle = (self.line[0] + self.line[1]) / 2
        self.size = np.round(np.abs(self.line[0] - self.line[1]))
        self.length = np.hypot(*self.size)
        if np.count_nonzero(self.size) != 2:
            if self.size[0] == 0:
                self.size[0] += 1
            else:
                self.size[1] += 1
