import matplotlib.pyplot as plt


def distance(p0, p1):
    #return math.sqrt((p0[0] - p1[0])*(p0[0] - p1[0]) + (p0[1] - p1[1])*(p0[1] - p1[1]))
    return abs(p0 - p1)


# Run from the top folder as:
# python3 -m dataset.lab_batch <args>
if __name__ == '__main__':
    import argparse


    input_dir = '../'
    input_file = 'output_colorfulness_run1.txt'

    # Argparse setup
    parser = argparse.ArgumentParser(
        description='Creates a graph visualization from the colorfulness data')
    parser.add_argument('-i', '--input',
                        default=input_dir + input_file,
                        type=str,
                        metavar='TXT FILE',
                        dest='input',
                        help='use TXT FILE as source of the input data '
                             '(default: {}) '
                        .format(input_dir))

    args = parser.parse_args()

        
    with open(args.input + input_file, 'r') as myfile:
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
