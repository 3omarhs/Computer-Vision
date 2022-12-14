import cv2
import numpy as np
import skimage.exposure

img = cv2.imread(r"C:\Users\Omar Hassan\PycharmProjects\Graduation project\gen3\photos\10photos_w_NE\Afton_Smith_0001.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = (40, 80, 80)
upper = (90, 255, 255)
mask = cv2.inRange(img_hsv, lower, upper)
mask = 255 - mask

kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Use initial, binary mask to set background to white
result = img.copy()
result[mask == 0] = (255, 255, 255)

# Save image just for intermediate output
cv2.imwrite('output_no_trans.png', result)

# Use antialiased mask as final alpha channel for transparency
mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)
mask = skimage.exposure.rescale_intensity(mask, in_range=(127.5, 255), out_range=(0, 255))
result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
result[:, :, 3] = mask

# Save final image
cv2.imwrite('output.png', result)