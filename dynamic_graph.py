import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from IPython import display
%matplotlib inline

for i in range(0,10) :
    plt.scatter(i, i**2)
    display.clear_output(wait=True)
    display.display(pl.gcf())
    time.sleep(1)
