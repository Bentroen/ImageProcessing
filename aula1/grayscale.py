import cv2
import numpy as np


img = cv2.imread("landscape.jpg")


width = img.shape[0]
height = img.shape[1]
channels = img.shape[2]

channel_blue = np.zeros(img.shape, dtype=np.uint8)
channel_green = np.zeros(img.shape, dtype=np.uint8)
channel_red = np.zeros(img.shape, dtype=np.uint8)

channel_blue[:,:,0] = img[:,:,0]
channel_green[:,:,1] = img[:,:,1]
channel_red[:,:,2] = img[:,:,2]

cv2.imshow("Blue channel", channel_blue)
cv2.imshow("Green channel", channel_green)
cv2.imshow("Red channel", channel_red)


# Generate grayscale image
channel_avg = np.mean(img[i][j])
cv2.imshow("Grayscale", channel_avg)
cv2.waitKey(0)


# Using for loop (less efficient)
grayscale2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        grayscale2[i][j] = img[i][j].sum() // 3
cv2.imshow("Grayscale 2", grayscale2)
cv2.waitKey(0)
