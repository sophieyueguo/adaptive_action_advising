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
            else:
                data[i][j] = [0, 0, 0] #not classified well
    
    if rotate:
        return np.rot90(np.array(data, dtype=np.uint8))
    return np.array(data, dtype=np.uint8)



student_X = np.load('world_model/student_X.npy', allow_pickle=True)
student_Y = np.load('world_model/student_Y.npy', allow_pickle=True)

# student_X = np.load('world_model/X.npy', allow_pickle=True)
# student_Y = np.load('world_model/rewardNN_Y.npy', allow_pickle=True)

i = 0
ind = 0
while i < 50:
    reward = student_Y[ind]
    if reward > 0:
        obs = student_X[ind][0]
        #obs = student_X[ind][0]['image']
        # print (obs)
        print ('sample', i, 'action is', student_X[ind][1])
        i += 1
        
        img = normalize_and_scale_rgb(obs, 1)
        plt.figure(figsize=(8, 4))
        plt.imshow(img)
        plt.savefig('world_model/debug/' + str(i) + '.png')
        # plt.show()
        plt.close()
        # print ('****************************************')
    ind += 1