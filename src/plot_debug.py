import matplotlib.pyplot as plt
import numpy as np

max_distance = 10

def reward_function(distance):
    if distance <=2:
        return 0.01 * distance
    else:
        return ((distance-2)/max_distance) ** 0.4

X = np.linspace(0, max_distance)
Y = [reward_function(x) for x in X]

# print(X)
# print(Y)
plt.plot(X,Y)
plt.show()
