import numpy as np
import matplotlib.pyplot as plt
import math

def function(x):
    safe_distance = 6
    if x >= safe_distance:
        return 1
    elif x <= 1:
        return -1
    else:
        return math.log(x+0.01) -1
max_distance = 10
X = np.linspace(0, max_distance)
Y = [function(x) for x in X]
plt.plot(X,Y)
plt.show()
