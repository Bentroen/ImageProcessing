from typing import Callable

import cv2
import numpy
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
    


if __name__ == "__main__":
    img = cv2.imread("rice.png")
    
    plot_histogram_grayscale(img)
    plot_histogram_channel(img, 0, "blue")
    plot_histogram_channel(img, 1, "green")
    plot_histogram_channel(img, 2, "red")