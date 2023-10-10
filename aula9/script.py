import os
import sys
from matplotlib import pyplot as plt
import numpy as np
import cv2

from aula8.script import add_padding, cross_kernel, erosion, full_kernel


KERNEL_SIZE = 5


def dilation(img: np.ndarray, kernel: np.ndarray):
    width, height = img.shape
    kernel_width, kernel_height = kernel.shape
    padding = kernel_width - 1
    padded_img = add_padding(img, padding // 2)
    dilation = np.zeros((width, height), dtype="float32")

    for i in range(width):
        for j in range(height):
            dilation[i][j] = 255 - (
                np.max(
                    np.multiply(
                        255 - padded_img[i : i + kernel_width, j : j + kernel_height],
                        kernel,
                    )
                )
                * 255
            )

    return dilation


def main():
    img = cv2.imread("aula9/challenge-accepted.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Make image black and white
    img[img < 100] = 0
    img[img >= 100] = 255

    # Prepare plot
    fig, axs = plt.subplots(2, 2)
    # Remove axis
    for ax in axs.flat:
        ax.set(xticks=[], yticks=[])

    # Original image
    axs[0][0].imshow(img, cmap="gray")
    axs[0][0].set_title("Original")

    # 3x3
    kernel = full_kernel(3)
    dilation_img = dilation(img, kernel)
    axs[0][1].imshow(dilation_img, cmap="gray")
    axs[0][1].set_title("Full 3x3")

    # 7x7
    kernel = full_kernel(7)
    dilation_img = dilation(img, kernel)
    axs[1][0].imshow(dilation_img, cmap="gray")
    axs[1][0].set_title("Full 7x7")

    # Circular 7x7
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilation_img = dilation(img, kernel)
    axs[1][1].imshow(dilation_img, cmap="gray")
    axs[1][1].set_title("Circular 7x7")

    plt.show()


if __name__ == "__main__":
    main()
