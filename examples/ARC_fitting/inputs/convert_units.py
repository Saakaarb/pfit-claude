import numpy as np

if __name__ == "__main__":

    data_filename = "NMC_SOC80_M2_normalized"

    # load filename and convert units

    with open(data_filename+".txt", "r") as f:
        data = np.genfromtxt(f, delimiter=" ")
    print(data.shape)
    # convert units
    data[:,0] = data[:,0] * 60 # to seconds
    data[:,1] = data[:,1]+273.15 # to Kelvin
    data[:,2] = data[:,2]/60 # K/s

    # save data
    with open(data_filename+"_converted_units.txt", "w") as f:
        np.savetxt(f, data, delimiter=",")

    print("Data converted and saved to", data_filename)
    