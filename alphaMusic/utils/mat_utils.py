import numpy as np

def uniform_in_range(ab, size):
    if len(ab) == 2:
        a, b = ab
        return (b-a) * np.random.random(size) + a
    elif len(ab) == 1:
        return ab * np.ones(size)
    else:
        raise ValueError(f'Range should be a list of 2 or 1 element, got {ab}')
        