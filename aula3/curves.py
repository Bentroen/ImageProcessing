from typing import Any, Callable
import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot_curves(func: Callable, *args: tuple[Any], title: str = ""):
    x = np.linspace(0, 255, 256)
    y = [func(value, *args) for value in x]

    plt.title(title)
    plt.plot(x, y)
    plt.show()


def apply_curve(img, func: Callable, args: tuple[Any]):
    width, height = img.shape
    new_img = np.zeros(img.shape, dtype="uint8")

    for i in range(width):
        for j in range(height):
            new_img[i][j] = func(img[i][j], *args)

    return new_img


def apply_curve_and_plot(img, func: Callable, *args: tuple[Any], title: str = ""):
    new_img = apply_curve(img, func, args)
    cv2.imshow(title, new_img)
    plot_curves(func, *args, title=title)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return new_img


def original(x):
    return x


def negative(x):
    return 255 - x


def s_curve(x, a: float = 1):
    x -= 128
    x /= 128
    return 255 / (1 + np.exp(-a * x))


def log_curve(x, a: float = 1):
    return np.log(1 + a * x)


def gamma_curve(x, gamma: float = 1):
    return 255 * np.power(x / 255, gamma)


def contrast(x, alpha: float = 1):
    return max(0, min(alpha * x, 255))


def crush_blacks(x, cutoff_point: int):
    return 0 if x < cutoff_point else (x - cutoff_point) * (255 / cutoff_point)


def crush_whites(x, cutoff_point: int):
    return 255 if x > cutoff_point else x * (255 / cutoff_point)


def main():
    img = cv2.imread("road.jpeg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    apply_curve_and_plot(img, original, title="Original")
    apply_curve_and_plot(img, negative, title="Negative")
    apply_curve_and_plot(img, s_curve, 5.0, title="S Curve")
    apply_curve_and_plot(img, log_curve, title="Log Curve")
    apply_curve_and_plot(img, gamma_curve, 4, title="Gamma Curve")
    apply_curve_and_plot(img, contrast, 1.5, title="Contrast")
    apply_curve_and_plot(img, crush_blacks, 128, title="Crush Blacks")
    apply_curve_and_plot(img, crush_whites, 128, title="Crush Whites")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
