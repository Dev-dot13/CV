import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read the input image
img = cv2.imread('pic6.jpg')  
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# 2. Define source points (corners of the object in the image)
# Example: corners of a tilted rectangle (update coordinates as per your image)
src_points = np.float32([[1150, 737], [1796, 840], [1196, 270], [1852, 337]])

# 3. Define destination points (desired rectangular output)
dst_points = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])

# 4. Compute the Perspective Transformation Matrix
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
print("Perspective Transformation Matrix:\n", matrix)

# 5. Apply the transformation
warped_img = cv2.warpPerspective(img_rgb, matrix, (400, 400))

# 6. Display the transformed image
plt.imshow(warped_img)
plt.title("Perspective Transformed Image")
plt.axis('off')
plt.show()