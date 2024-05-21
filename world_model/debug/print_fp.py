import numpy as np
import matplotlib.pyplot as plt


def normalize_and_scale_rgb(raw_data, normalization=1, rotate=True):
    """
    Normalize and scale RGB values in the data.
    """
    m, n = len(raw_data), len(raw_data[0])
    data = [[0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):

            # value = round(raw_data[i][j][0] * normalization)
            value = raw_data[i][j][0] * normalization
            if value == 1:
                data[i][j] = [255, 255, 255]  # White
            elif value == 2:
                data[i][j] = [128, 128, 128]  # Grey
            elif value == 3:
                data[i][j] = [255, 255, 255]  # room entrance, same as empty space Red
            elif value == 5:
                data[i][j] = [0, 0, 255] # key
                # print ('key', i, j)
            elif value == 8:
                data[i][j] = [0, 255, 0]  # Green
                # print ('goal', i, j)
            elif value == 10:
                data[i][j] = [255, 0, 0]  # red
                # print ('agent', i, j)
                print ('direction', raw_data[i][j][2])
                dir = raw_data[i][j][2]
                dir_c = [255, 192, 203]
                if dir == 0:
                    data[i+1][j] = dir_c
                elif dir == 1:
                    data[i][j+1] = dir_c
                elif dir == 2:
                    data[i-1][j] = dir_c
                elif dir == 3:
                    data[i][j-1] = dir_c
            # else:
            #     data[i][j] = [0, 0, 0] #not classified well
    
    if rotate:
        return np.rot90(np.array(data, dtype=np.uint8))
    return np.array(data, dtype=np.uint8)



fp_pairs = np.load('world_model/debug_fp/fp_pairs.npy', allow_pickle=True)
for i in range(len(fp_pairs)):
    obs = fp_pairs[i][0]
    print ('sample', i, 'action', fp_pairs[i][1])
    img = normalize_and_scale_rgb(obs, 1)
    plt.figure(figsize=(8, 4))
    plt.imshow(img)
    plt.savefig('world_model/debug_fp/' + str(i) + '.png')
    # plt.show()
    plt.close()
    print ('****************************************')
