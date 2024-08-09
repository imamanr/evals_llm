import numpy as np
from numpy.linalg import norm


def cosineSim(A,B):
    return np.dot(A,B)/(norm(A)*norm(B))
