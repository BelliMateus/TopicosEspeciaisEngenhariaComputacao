import cv2
import numpy as np
from matplotlib import pyplot as plt

img_c = cv2.imread('lena_clean.jpg',0)
img_n = cv2.imread('lena_noise.jpg',0)

laplacian_c = cv2.Laplacian(img_c,cv2.CV_64F)
laplacian_n = cv2.Laplacian(img_n,cv2.CV_64F)

plt.subplot(2,2,1),plt.imshow(img_c,cmap = 'gray')
plt.title('Original sem Ruido'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian_c,cmap = 'gray')
plt.title('Laplaciano do Original sem Ruido'), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3),plt.imshow(img_n,cmap = 'gray')
plt.title('Original com Ruido'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(laplacian_n,cmap = 'gray')
plt.title('Laplaciano do Original com Ruido Clean'), plt.xticks([]), plt.yticks([])

plt.show()