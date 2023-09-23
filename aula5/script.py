import numpy as np
import cv2
import matplotlib.pyplot as plt


def histogram_equalization(img):
    # Histograma da imagem
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # CDF da imagem
    cdf = hist.cumsum()

    # Normalização do CDF
    cdf_normalized = cdf * hist.max() / cdf.max()

    # Equalização
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")
    img2 = cdf[img]

    # OU:
    # img2 = cv2.equalizeHist(img)

    # Histograma da imagem equalizada
    hist2, bins2 = np.histogram(img2.flatten(), 256, [0, 256])

    # CDF da imagem equalizada
    cdf2 = hist2.cumsum()

    # Normalização do CDF da imagem equalizada
    cdf_normalized2 = cdf2 * hist2.max() / cdf2.max()

    # Plotagem dos histogramas
    plt.plot(cdf_normalized, color="b")
    plt.plot(cdf_normalized2, color="r")
    plt.hist(img.flatten(), 256, [0, 256], color="lightblue")
    plt.hist(img2.flatten(), 256, [0, 256], color="pink")
    plt.xlim([0, 256])
    plt.legend(
        (
            "CDF Original",
            "CDF Equalizado",
            "Histograma Original",
            "Histograma Equalizado",
        ),
        loc="upper left",
    )
    plt.show()

    # Plotagem das imagens
    plt.subplot(121), plt.imshow(img, cmap="gray")
    plt.title("Imagem Original"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap="gray")
    plt.title("Imagem Equalizada"), plt.xticks([]), plt.yticks([])
    plt.show()


def main():
    img = cv2.imread("leaf.jpg")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    histogram_equalization(gray)


if __name__ == "__main__":
    main()
