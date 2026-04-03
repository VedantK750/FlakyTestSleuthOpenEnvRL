import random


def unstable_sum(values):
    random.shuffle(values)
    return values[0] + values[1]
