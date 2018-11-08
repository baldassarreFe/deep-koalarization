import numpy as np
import matplotlib.pyplot as plt

'''
data = np.genfromtxt('output_train_time_per_batch_1.txt', delimiter=',')
print(data)
for i in range(len(data)):
    print(str(data[i]))
'''

# Run from the top folder as:
# python3 -m dataset.lab_batch <args>
if __name__ == '__main__':
    import argparse


    input_dir = '../'
    input_file = 'output_train_time_per_batch_run1.txt'

    # Argparse setup
    parser = argparse.ArgumentParser(
        description='Creates a graph visualization from the train_time_per_batch data')
    parser.add_argument('-i', '--input',
                        default=input_dir + input_file,
                        type=str,
                        metavar='TXT FILE',
                        dest='input',
                        help='use TXT FILE as source of the input data '
                             '(default: {}) '
                        .format(input_dir))

    args = parser.parse_args()

    with open (args.input + input_file, "r") as myfile:
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
