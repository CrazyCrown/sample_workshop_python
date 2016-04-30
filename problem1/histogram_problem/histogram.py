# coding: utf-8
"""Sample Histogram

@author: Liz
@modified: 04-29-2016

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sample.jpg', 0)

# Analysis
hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='g')
plt.xlim([0, 256])
plt.legend(('cdf', 'histogram'), loc='upper left')
plt.show()


# Compare the both grayscale and histogram
equ = cv2.equalizeHist(img)
res = np.hstack((img, equ))
cv2.imwrite('histogram_res_test.jpg', res)
