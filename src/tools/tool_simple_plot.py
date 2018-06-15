import numpy as np
import matplotlib.pyplot as plt
import math

def function(x):
        # safe_distance = 20
        # if x >= safe_distance:
        #     return math.log(safe_distance+0.01) -1
        # elif x <= 1:
        #     return -1
        # else:
        #     return math.log(x+0.01) -1
        max_distance = 73
        return (1 - x / max_distance)

# max_distance = 20
max_distance = 73
X = np.linspace(0, max_distance)
Y = [function(x) for x in X]
plt.plot(X,Y)
# plt.xlabel("Distance avec l'ennemi le plus proche")
plt.xlabel("Distance de l'agent vis à vis de la nourriture")
plt.ylabel("Récompense")
plt.show()
