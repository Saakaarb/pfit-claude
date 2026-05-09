from matplotlib import pyplot as plt

import numpy as np

if __name__ == "__main__":
    data_filename = "NMC_SOC80_M2_normalized_converted_units"
    data = np.genfromtxt(data_filename+".csv", delimiter=",")
    plt.plot(data[:,0], data[:,1])
    plt.show()
    plt.close()

    plt.plot(data[:,1], data[:,2])
    plt.yscale("log")
    plt.show()
    plt.close()
