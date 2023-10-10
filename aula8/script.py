import numpy as np
import cv2
import matplotlib.pyplot as plt


KERNEL_SIZE = 5


def full_kernel(size: int = 3):
    return np.ones((size, size), dtype=np.float32)


def cross_kernel(size: int = 3):
    kernel = np.ones((size, size), dtype=np.float32)
    kernel[size // 2] = np.zeros(size, dtype=np.float32)
    kernel[:, size // 2] = np.zeros(size, dtype=np.float32)
    return kernel


def add_padding(img: np.ndarray, padding: int = 1):
    width, height = img.shape
    padded_img = np.zeros((width + padding * 2, height + padding * 2), dtype=np.uint8)
    for i in range(width):
        for j in range(height):
            padded_img[i + padding][j + padding] = img[i][j]
    return padded_img


def erosion(img: np.ndarray, kernel: np.ndarray):
    width, height = img.shape
    kernel_width, kernel_height = kernel.shape
    padding = kernel_width - 1
    padded_img = add_padding(img, padding // 2)
    erosion = np.zeros((width + padding, height + padding), dtype="float32")

    for i in range(width):
        for j in range(height):
            erosion[i][j] = (
                0
                if np.all(
                    np.multiply(
                        padded_img[i : i + kernel_width, j : j + kernel_height], kernel
                    )
                )
                == 0
                else 255
            )

    erosion = np.clip(erosion, 0, 255)
    erosion = erosion.astype(np.uint8)

    return erosion[padding // 2 : -padding // 2, padding // 2 : -padding // 2]


def main():
    img = cv2.imread("circles.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cross_kernel(KERNEL_SIZE)
    kernel = full_kernel(KERNEL_SIZE)

    fig, axs = plt.subplots(3, 3)
    # Remove axis
    for ax in axs.flat:
        ax.set(xticks=[], yticks=[])
    axs[0][0].imshow(img, cmap="gray")
    axs[0][0].set_title("Original")

    for i in range(8):
        kernel = np.array(kernel, dtype=np.float32)
        print(f"Step {i+1}")
        img = erosion(img, kernel)
        x, y = divmod(i + 1, 3)
        axs[x][y].imshow(img, cmap="gray")
        axs[x][y].set_title(f"Erosion {i + 1}")

    plt.show()


if __name__ == "__main__":
    main()
