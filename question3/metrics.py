#METRICS.PY
import math
import numpy as np
def accuracy(y_hat, y):
    assert(y_hat.size == y.size)
    return((y_hat==y).mean())
    # TODO: Write here
