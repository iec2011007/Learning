__author__ = 'root'


import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

np.random.seed(0)

noisy_moon = datasets.make_moons(n_samples=1500, noise = 0.05)
print noisy_moon.__class__