import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Read and display original image
img = cv2.imread('pic3.jpg')     
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

h, w = img.shape[:2]

# 2. Translation (shift image)
tx, ty = 200, 300  # shift by 50px right and 30px down
T = np.float32([[1, 0, tx],
                [0, 1, ty]])
translated = cv2.warpAffine(img_rgb, T, (w, h))

plt.imshow(translated)
plt.title("Translated Image (200px right, 300px down)")
plt.axis('off')
plt.show()

# 3. Rotation
M = cv2.getRotationMatrix2D((w/2, h/2), 45, 1)
rotated = cv2.warpAffine(img_rgb, M, (w, h))

plt.imshow(rotated)
plt.title("Rotated Image (45°)")
plt.axis('off')
plt.show()

# 4. Scaling
scaled = cv2.resize(img_rgb, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

plt.imshow(scaled)
plt.title("Scaled Image (1.5x)")
plt.axis('off')
plt.show()

# 5. Reflection
reflected = cv2.flip(img_rgb, 1)

plt.imshow(reflected)
plt.title("Reflected Image (Horizontal Flip)")
plt.axis('off')
plt.show()

# 6. Shearing
shear_matrix = np.float32([[1, 0.3, 0],
                           [0.1, 1, 0]])
sheared = cv2.warpAffine(img_rgb, shear_matrix, (int(w*1.5), int(h*1.5)))

plt.imshow(sheared)
plt.title("Sheared Image")
plt.axis('off')
plt.show()