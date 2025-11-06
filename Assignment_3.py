import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('pic5.jpg')  
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# 2. Define source and destination points (manually chosen or from features)
# These are example corner coordinates of a document or rectangular object in the image
src_points = np.float32([[41, 228], [352, 120], [444, 370], [767, 237]])

# Destination coordinates (desired straightened rectangle)
dst_points = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])

# 3. Compute the Homography Matrix
H, status = cv2.findHomography(src_points, dst_points)
print("Homography Matrix (H):\n", H)

# 4. Apply perspective warp using the computed matrix
warped_img = cv2.warpPerspective(img_rgb, H, (400, 400))

# 5. Display the transformed image
plt.imshow(warped_img)
plt.title("Perspective Corrected (Warped) Image")
plt.axis('off')
plt.show()