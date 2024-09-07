import h5py
import numpy as np

# Open the H5 file in read mode
with h5py.File('./prepro_data/resnet14x14.h5', 'r') as file:
    print("Keys: %s" % file.keys())
    a_group_key = list(file.keys())[0]
    
    # Getting the data
    data = np.array(file[a_group_key])
    print(data)
    
    # Write data to a text file
    with open('featuremap.txt', 'w') as f:
        for array in data:
            for row in array:
                f.write(" ".join(map(str, row)) + "\n")
