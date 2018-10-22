import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('../output_cost_run1.txt')
print(data)

plt.grid()
plt.plot(data)
plt.xlabel('img')
plt.ylabel('cost')
plt.show()
