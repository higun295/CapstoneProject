import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 읽기 (Grayscale)
img = cv2.imread('Lenna.png', cv2.IMREAD_GRAYSCALE)

# 히스토그램 계산
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 히스토그램 출력
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()