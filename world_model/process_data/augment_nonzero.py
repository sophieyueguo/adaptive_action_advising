import numpy as np
X = np.load('world_model/X_nonzero_reward.npy', allow_pickle=True)
Y = np.load('world_model/rewardNN_Y_nonzero_reward.npy', allow_pickle=True)

X_augment = []
Y_augment = []

change_dir_cnt = 0
change_action_cnt = 0

n = len(X)
for i in range(int(n/3)):

    change_dir = False
    change_action = False

    img = X[i][0]['image']
    action_mask = X[i][0]['action_mask']
    action = X[i][1]
    for r in range(len(img)):
        for c in range(len(img[0])):
            if img[r][c][0] == 10:
                dir = img[r][c][2]
                new_dir = np.random.randint(4) # random direction
                if new_dir != dir:
                    change_dir = True
                    img[r][c][2] = new_dir
                    change_dir_cnt += 1
                    new_action = action

    # random action
    if not change_dir:
        new_action = np.random.randint(len(action_mask))
        if new_action != action and action_mask[new_action] == 1:
            change_action = True
            change_action_cnt += 1

    if change_dir or change_action:
        X_augment.append([{'image': img, 'action_mask': action_mask}, new_action])
        Y_augment.append(0)


# print (X_augment[0])
print ('length', len(X_augment), len(Y_augment))
print ('change_dir_cnt', change_dir_cnt, 'change_action_cnt', change_action_cnt)

np.save('world_model/X_augment.npy', X_augment)
np.save('world_model/rewardNN_Y_augment.npy', Y_augment)

