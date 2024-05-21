import numpy as np

# Load the two .npy files
for filetype in ['X', 'rewardNN_Y']:

    array1 = np.load('world_model/' + filetype + '_expert.npy', allow_pickle=True)
    array2 = np.load('world_model/' + filetype + '_random_reward.npy', allow_pickle=True)

    # Print the shapes of the original arrays
    print("Shape of array1:", array1.shape)
    print("Shape of array2:", array2.shape)

    # Concatenate the arrays
    # axis=0 will concatenate along the first dimension
    # Change the axis parameter if you need to concatenate along a different dimension
    concatenated_array = np.concatenate((array1, array2), axis=0)

    # Print the shape of the concatenated array
    print("Shape of concatenated array:", concatenated_array.shape)

    # Save the concatenated array to a new .npy file
    np.save('world_model/' + filetype + '.npy', concatenated_array)
