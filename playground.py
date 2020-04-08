# # import numpy as np
# #
# # # print(range(5))
# # #
# # # for i in range(5):
# # #     print(i)
# #
# # a = np.array([[629, 361], [1066, 411], [396, 486], [946, 571]])
# #
# # x = a[:, 0]
# #
# # print(x)
import numpy as np

#
# from functions import hits, fits, convert_mask
#
im = np.array([[4, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 4]])

print(np.average(im))

print(im[3:5, 1:2])
