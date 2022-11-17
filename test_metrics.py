from mmhuman3d.core.evaluation import keypoint_mpjpe
import numpy as np

gt = np.random.rand(9, 6, 3)
pred = np.copy(gt)
mask = np.ones((9, 6), dtype=bool)

# mpjpe
print(keypoint_mpjpe(pred, gt, mask, alignment='none'))

rotate = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

pred = np.dot(pred, rotate)

# mpjpe
print(keypoint_mpjpe(pred, gt, mask, alignment='none'))
# pa-mpjpe
print(keypoint_mpjpe(pred, gt, mask, alignment='procrustes'))


# 2D
gt = np.random.rand(8, 5, 2)
pred = np.copy(gt)
mask = np.ones((8, 5), dtype=bool)
rotate = np.array([[1, 0], [0, -1]])
pred = np.dot(pred, rotate)
# mpjpe
print("2D mpjpe", keypoint_mpjpe(pred, gt, mask, alignment='none'))
# pa-mpjpe
print("2D pa-mpjpe", keypoint_mpjpe(pred, gt, mask, alignment='scale'))
