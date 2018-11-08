import numpy as np
import matplotlib.pyplot as plt

# Run from the top folder as:
# python3 -m dataset.lab_batch <args>
if __name__ == '__main__':
    import argparse


    input_dir = '../'
    input_file = 'output_cost_run1.txt'

    # Argparse setup
    parser = argparse.ArgumentParser(
        description='Creates a graph visualization from the cost data')
    parser.add_argument('-i', '--input',
                        default=input_dir + input_file,
                        type=str,
                        metavar='TXT FILE',
                        dest='input',
                        help='use TXT FILE as source of the input data '
                             '(default: {}) '
                        .format(input_dir))

    args = parser.parse_args()

    data = np.genfromtxt(args.input + input_file)
    print(data)

    plt.grid()
    plt.plot(data)
    plt.xlabel('img')
    plt.ylabel('cost')
    plt.show()
