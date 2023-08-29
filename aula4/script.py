import numpy as np
import cv2
import matplotlib.pyplot as plt


def histogram_expansion(img, r1: int, r2: int):
    width, height = img.shape
    new = np.zeros((width, height), dtype="uint8")
    for i in range(width):
        for j in range(height):
            new_value = 255 * ((img[i, j] - r1) / (r2 - r1))
            new_value = max(0, min(new_value, 255))
            new[i, j] = new_value
    return new


def histogram_compression(img, r1: int, r2: int):
    pass


def plot_histogram(img, title: str = ""):
    width, height = img.shape
    histogram = [0] * 256
    new = np.zeros((width, height))
    for i in range(width):
        for j in range(height):
            pixel = int(max(0, min(img[i, j], 255)))
            if histogram[pixel] < 5000:
                histogram[pixel] += 1
    plt.title(title)
    plt.bar(range(256), histogram)
    plt.show()


def main():
    img = cv2.imread("snow.jpg")
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

    # Convert to grayscale
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Original", grayscale)
    plot_histogram(grayscale, "Original")
    
    expanded = histogram_expansion(grayscale, 100, 200)
    cv2.imshow("Expanded", expanded)
    plot_histogram(expanded, "Expanded")
    
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
