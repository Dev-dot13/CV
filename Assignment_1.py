import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1. Read the image
img = cv2.imread('pic3.jpg')  
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB

# 2. Display original image
plt.figure(figsize=(5,5))
plt.title("Original Image")
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

# 3. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# 4. Resize image
resized = cv2.resize(img_rgb, (100, 100))
plt.imshow(resized)
plt.title("Resized Image (100x100)")
plt.axis('off')
plt.show()

# 5. Apply Thresholding
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
plt.imshow(thresh, cmap='gray')
plt.title("Thresholded Image")
plt.axis('off')
plt.show()

# 6. Apply Gaussian Blur (Filtering)
blurred = cv2.GaussianBlur(img_rgb, (7,7), 0)
plt.imshow(blurred)
plt.title("Gaussian Blurred Image")
plt.axis('off')
plt.show()

# 7. Save processed images
cv2.imwrite('grayscale.jpg', gray)
cv2.imwrite('thresholded.jpg', thresh)
cv2.imwrite('blurred.jpg', cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))

print("✅ Images saved successfully!")