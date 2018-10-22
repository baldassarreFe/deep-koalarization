import matplotlib.pyplot as plt


def distance(p0, p1):
    #return math.sqrt((p0[0] - p1[0])*(p0[0] - p1[0]) + (p0[1] - p1[1])*(p0[1] - p1[1]))
    return abs(p0 - p1)


with open('../output_colorfulness_run1.txt', 'r') as myfile:
    data = myfile.readlines()
    print(data)

# convert string list into int list of milliseconds
new_data = []
for i in range(len(data)):
    d = data[i]
    d = d.split(',')
    new_data.append([float(d[0]), float(d[1])])

# compute diff between output and groundtruth
dist = []
for i in range(len(new_data)):
    d = new_data[i]
    dist.append(distance(d[0], d[1]))

plt.figure(1)
plt.subplot(211)
plt.grid()
plt.plot(new_data)
plt.ylabel('score')
plt.legend(['output', 'groundtruth'])

plt.subplot(212)
plt.grid()
plt.plot(dist)
plt.xlabel('img')
plt.ylabel('error')
plt.legend(['img', 'error'])
plt.show()
