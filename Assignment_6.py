import cv2
import numpy as np

# Step 1: Example corresponding points (you can replace with actual matched points)
pts1 = np.float32([[100,150],[120,130],[150,200],[180,160],[200,220],[250,210],[300,250],[320,300]])
pts2 = np.float32([[102,148],[118,128],[152,198],[182,158],[198,218],[248,208],[298,248],[318,298]])

# Step 2: Compute Fundamental Matrix using 8-point algorithm
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)

print("Fundamental Matrix (F):\n", F)

# Step 3: Validate Epipolar Constraint  (x2^T * F * x1 ≈ 0)
pts1_h = cv2.convertPointsToHomogeneous(pts1).reshape(-1,3)
pts2_h = cv2.convertPointsToHomogeneous(pts2).reshape(-1,3)

for i in range(len(pts1)):
    val = np.dot(pts2_h[i], np.dot(F, pts1_h[i]))
    print(f"Epipolar Constraint for point {i+1}: {val:.4f}")