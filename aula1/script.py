import cv2
import numpy


img = cv2.imread("landscape.jpg")


print("Largura em pixels: ", img.shape[1])
print("Altura em pixels: ", img.shape[2])

print("Quantidade de canais: ", img.shape[2])


# O pixel superior mais à esquerda possui coordenadas (0, 0)
(b, g, r) = img[0, 0]  # ordem do OpenCV é BGR e não RGB


cv2.imshow("Paisagem", img)
cv2.waitKey(0) # espera pressionar qualquer tecla


# Mostrar apenas o canal azul
cv2.imshow("Canal azul", img[:, :, 0])
cv2.imshow("Canal verde", img[:, :, 1])
cv2.imshow("Canal vermelho", img[:, :, 2])
cv2.waitKey(0)


cv2.imwrite("output.jpg", img)
