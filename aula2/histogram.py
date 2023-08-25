from typing import Callable

import cv2
import numpy as np
import matplotlib.pyplot as plt



def plot_histogram(img, func: Callable, plt_color: str):
    width, height, _ = img.shape
    
    histogram = [0]* 256
    for i in range(width):
        for j in range(height):
            value = func(img[i][j])
            histogram[value] += 1
    
    plt.title(f"Histogram ({plt_color} channel)")
    plt.xlabel("Pixel value")
    plt.ylabel("Amount")
    plt.bar(range(256), histogram, color=plt_color)
    plt.show()
    

def plot_histogram_grayscale(img):
    plot_histogram(img, lambda x: x.sum() // 3, "gray")
    
    
def plot_histogram_channel(img, channel: int, plt_color: str):
    plot_histogram(img, lambda x: x[channel], plt_color)
    

def filter_level(img, func: Callable[[int], bool], replace_with: int):
    width, height, _ = img.shape
    new_img = np.zeros(img.shape, dtype="uint8")
    
    for i in range(width):
        for j in range(height):
            value = np.mean(img[i][j])
            if func(value):
                new_img[i][j] = replace_with
            else:
                new_img[i][j] = img[i][j]
                
    return new_img


if __name__ == "__main__":
    img = cv2.imread("rice.png")
    
    plot_histogram_grayscale(img)
    plot_histogram_channel(img, 0, "blue")
    plot_histogram_channel(img, 1, "green")
    plot_histogram_channel(img, 2, "red")
    
    cutoff_point = 160
    
    img_rice = filter_level(img, lambda x: x >= cutoff_point, 255)
    img_beans = filter_level(img, lambda x: x < cutoff_point, 255)
    
    cv2.imshow("Rice and beans", img)
    cv2.imshow("Rice", img_rice)
    cv2.imshow("Beans", img_beans)
    cv2.imshow("Paisagem", img)
    cv2.waitKey(0)
