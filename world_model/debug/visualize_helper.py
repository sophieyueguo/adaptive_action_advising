import numpy as np
import matplotlib.pyplot as plt

from train_transNN import normalize_and_scale_rgb



# output = np.array([ 2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,
# 0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,
# 5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,
# 2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  1.,  0.,
# 0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,
# 0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,
# 1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,
# 0.,  2.,  5.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,
# 0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,
# 2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,
# 0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  2.,
# 5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,
# 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,
# 0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,
# 0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,
# 3.,  2.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  3.,  2.,
# 0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  3.,  2.,  0.,  2.,
# 5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  3.,  2.,  0.,  2.,  5.,  0.,
# 2.,  5.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,
# 0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,
# 5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,
# 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  2.,  5.,
# 0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,
# 0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,
# 1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,
# 0.,  1.,  0.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,
# 0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,
# 1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,
# 0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,
# 5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  3.,  2.,  0.,  2.,  5.,  0.,
# 2.,  5.,  0.,  2.,  5.,  0.,  3.,  2.,  0.,  2.,  5.,  0.,  2.,  5.,
# 0.,  2.,  5.,  0.,  3.,  2.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,
# 5.,  0.,  3.,  2.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,
# 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  3.,  2.,  0.,  1.,  0.,
# 0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,
# 0.,  0.,  1.,  0.,  0.,  3.,  2.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,
# 1.,  0.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  5.,  0.,
# 0., 10.,  0.,  3.,  3.,  2.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,
# 0.,  0.,  3.,  2.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,
# 3.,  2.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,
# 0.,  2.,  5.,  0.,  8.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  3.,
# 2.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,
# 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  3.,  2.,  0.,  1.,  0.,
# 0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,
# 5.,  0.,  3.,  2.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,
# 3.,  2.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  3.,  2.,
# 0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  3.,  2.,  0.,  2.,
# 5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,
# 1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,
# 0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,
# 5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,
# 2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,
# 0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,
# 0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,
# 1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  1.,  0.,
# 0.,  1.,  0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,
# 0.,  0.,  1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,
# 1.,  0.,  0.,  2.,  5.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,
# 0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,
# 5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,
# 2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,
# 0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.,  2.,  5.,  0.])

# obs = output.reshape(17, 17, 3)




obs = np.array([[[ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 2,  5,  0],
        [ 3,  2,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 3,  2,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 3,  2,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 3,  2,  0],
        [ 2,  5,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 2,  5,  0],
        [ 3,  2,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 3,  2,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 3,  2,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 3,  2,  0],
        [ 2,  5,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 3,  2,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 3,  2,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 3,  2,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 3,  2,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 3,  2,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 3,  2,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 3,  2,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 2,  5,  0],
        [ 3,  2,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 3,  2,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 3,  2,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 3,  2,  0],
        [ 2,  5,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [10,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 8,  1,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 1,  0,  0],
        [ 2,  5,  0]],

       [[ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0],
        [ 2,  5,  0]]])

img = normalize_and_scale_rgb(obs, 1)
# np.save('test', output)
plt.figure(figsize=(8, 4))
plt.imshow(img)
plt.savefig('test.png')
plt.show()
plt.close()
print ('****************************************')
print ('saved a test figure')