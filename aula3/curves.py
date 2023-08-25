from typing import Any, Callable
import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot_curves(img, func: Callable, args: tuple[Any], title: str = ""):
    x = np.linspace(0, 255, 256)
    y = func(x, *args)

    plt.plot(x, y)
    plt.show()


def _process_image(img, func: Callable, args: tuple[Any]):
    width, height = img.shape
    new_img = np.zeros(img.shape, dtype="uint8")

    for i in range(width):
        for j in range(height):
            new_img[i][j] = func(img[i][j], *args)

    return new_img


def negative(img):
    width, height = img.shape
    new_img = np.zeros(img.shape, dtype="uint8")

    for i in range(width):
        for j in range(height):
            new_img[i][j] = 255 - img[i][j]

    # Line plot of tone curve
    plt.plot(range(256), range(256)[::-1])
    plt.show()

    return new_img


def s_curve(img):
    width, height = img.shape
    new_img = np.zeros(img.shape, dtype="uint8")

    for i in range(width):
        for j in range(height):
            value = img[i][j]
            new_value = 255 / (1 + np.exp(-value))
            new_img[i][j] = new_value

    return new_img


def log_curve(img):
    width, height = img.shape
    new_img = np.zeros(img.shape, dtype="uint8")

    for i in range(width):
        for j in range(height):
            value = img[i][j]
            new_value = np.log(1 + value)
            new_img[i][j] = new_value

    return new_img


def gamma_curve(img, gamma: float):
    width, height = img.shape
    new_img = np.zeros(img.shape, dtype="uint8")

    for i in range(width):
        for j in range(height):
            value = img[i][j]
            new_value = 255 * np.power(value / 255, gamma)
            new_img[i][j] = new_value

    return new_img


def contrast(img, alpha: float):
    width, height = img.shape
    new_img = np.zeros(img.shape, dtype="uint8")

    for i in range(width):
        for j in range(height):
            value = img[i][j]
            new_value = alpha * value
            new_img[i][j] = new_value

    return new_img


def crush_blacks(img, cutoff_point: int):
    width, height = img.shape
    new_img = np.zeros(img.shape, dtype="uint8")

    for i in range(width):
        for j in range(height):
            value = img[i][j]
            if value < cutoff_point:
                new_img[i][j] = 0
            else:
                new_img[i][j] = value * 255 / cutoff_point

    return new_img


def crush_whites(img, cutoff_point: int):
    width, height = img.shape
    new_img = np.zeros(img.shape, dtype="uint8")

    for i in range(width):
        for j in range(height):
            value = img[i][j]
            if value > cutoff_point:
                new_img[i][j] = 255
            else:
                new_img[i][j] = value * 255 / cutoff_point

    return new_img


def main():
    img = cv2.imread("road.jpeg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    img_negative = negative(img)
    img_s_curve = s_curve(img)
    img_log_curve = log_curve(img)
    img_gamma_curve = gamma_curve(img, 0.5)
    img_contrast = contrast(img, 1.5)
    img_crush_blacks = crush_blacks(img, 128)
    img_crush_whites = crush_whites(img, 128)

    cv2.imshow("Original", img)
    cv2.imshow("Negative", img_negative)
    # cv2.imshow("S Curve", img_s_curve)
    # cv2.imshow("Log Curve", img_log_curve)
    cv2.imshow("Gamma Curve", img_gamma_curve)
    cv2.imshow("Contrast", img_contrast)
    cv2.imshow("Crush Blacks", img_crush_blacks)
    cv2.imshow("Crush Whites", img_crush_whites)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
