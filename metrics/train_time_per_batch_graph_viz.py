import numpy as np
import matplotlib.pyplot as plt

'''
data = np.genfromtxt('output_train_time_per_batch_1.txt', delimiter=',')
print(data)
for i in range(len(data)):
    print(str(data[i]))
'''

with open ("../output_train_time_per_batch_run1.txt", "r") as myfile:
    data=myfile.readlines()
    print(data)

# convert string list into int list of milliseconds
new_data = []
for i in range(len(data)):
    if data[i] != '\n':
        d = data[i]
        d = d.split('.')
        new_data.append(int(d[1]))

print(new_data)

plt.grid()
plt.plot(new_data)
plt.xlabel('batch')
plt.ylabel('milliseconds')
plt.show()
