import numpy as np
import cv2
import matplotlib.pyplot as plt


def convolution(img: np.ndarray, kernel: np.ndarray):
    width, height = img.shape
    kernel_width, kernel_height = kernel.shape
    padding = kernel_width - 1
    padded_img = add_padding(img, padding // 2)
    convolution = np.zeros((width + padding, height + padding), dtype="float32")

    for i in range(width):
        for j in range(height):
            convolution[i][j] = np.sum(
                np.multiply(
                    padded_img[i : i + kernel_width, j : j + kernel_height], kernel
                )
            )

    convolution = np.clip(convolution, 0, 255)
    convolution = convolution.astype(np.uint8)

    return convolution[padding // 2 : -padding // 2, padding // 2 : -padding // 2]


def add_padding(img: np.ndarray, padding: int = 1):
    width, height = img.shape
    padded_img = np.zeros((width + padding * 2, height + padding * 2), dtype=np.uint8)
    for i in range(width):
        for j in range(height):
            padded_img[i + padding][j + padding] = img[i][j]
    return padded_img


def main():
    img = cv2.imread("bridge.jpg")
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    padded = add_padding(gray)

    # Apply convolution
    kernel = [
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
    ]

    kernel_array = np.array(kernel, dtype=np.float32)
    conv = convolution(gray, kernel_array)

    # Show image
    cv2.imshow("Original image", gray)
    cv2.imshow("Convolution", conv)
    cv2.waitKey(0)

    # Repeat for n steps
    new_img = gray
    for i in range(5):
        new_img = convolution(new_img, kernel)
        cv2.imshow(f"Step {i+1}", new_img)


if __name__ == "__main__":
    main()
