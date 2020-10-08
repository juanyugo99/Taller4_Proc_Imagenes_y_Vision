import cv2
import math
import numpy as np


# Ambos metodos cumplen con la funión de detectar el click del mouse
def mouse_click_detection1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN: # Si el mouse es clickeado
        cv2.circle(image1, (x, y), 3, (255, 255, 255), -1) # dibuja un circulo en la posición del click
        points1.append((x, y)) # guarda la posición del click


def mouse_click_detection2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image2, (x, y), 3, (255, 255, 255), -1)
        points2.append((x, y))

# Definición de los puntos y de la primera imagen (lena.png)
points1 = []
path_file = input('Inserte la ruta de la imagen 1: ')
image1 = cv2.imread(path_file)
cv2.namedWindow("Lena")
cv2.setMouseCallback("Lena", mouse_click_detection1)

# Se muestra la imagen y espera a que el usuario clickee 3 veces en ella
while True:
    cv2.imshow('Lena', image1)
    k = cv2.waitKey(1) & 0xFF
    if (k == 27) or (len(points1) == 3):
        break
cv2.destroyAllWindows()

# Definición de los puntos y de la segunda imagen (lena_warped.png)
points2 = []
path_file = input('Inserte la ruta de la imagen 2: ')
image2 = cv2.imread(path_file)
cv2.namedWindow("Lena Warped")
cv2.setMouseCallback("Lena Warped", mouse_click_detection2)

# Se muestra la imagen y espera a que el usuario clickee 3 veces en ella
while True:
    cv2.imshow('Lena Warped', image2)
    k = cv2.waitKey(1) & 0xFF
    if (k == 27) or (len(points2) == 3):
        break
cv2.destroyAllWindows()

# Print de los 6 puntos tomados de las dos imagenes
print("p1: {} y p2: {}".format(points1, points2))

# A partir de los seis puntos se hace la transformada afín a la primera imagen
pts1 = np.float32(points1)
pts2 = np.float32(points2)
M_affine = cv2.getAffineTransform(pts1, pts2)
image_affine = cv2.warpAffine(image1, M_affine, image1.shape[:2])
compare = cv2.hconcat([image1, image2, image_affine])
# se muestran las dos iamgenes cargadas y el resultado de la transformación
cv2.imshow("Compare: normal image, image warped & image affine", compare)

# a partir de la matriz afín se obtienen los parametros de traslación, escalamiento y rotación
# según lo descrito por el ingeniero H. Rhody del Chester F. Carlson Center for imaging Science en su Lecture #2
# Septiembre 8, 2005

# parametros de escala
s0 = np.sqrt(M_affine[0, 0] ** 2 + M_affine[1, 0] ** 2)
s1 = np.sqrt(M_affine[0, 1] ** 2 + M_affine[1, 1] ** 2)
# parametro de rotación
theta = -np.arctan(M_affine[1, 0] / M_affine[0, 0])
# parametros de traslación
x0 = (M_affine[0, 2] * np.cos(theta) - M_affine[1, 2] * np.sin(theta)) / s0
x1 = (M_affine[0, 2] * np.sin(theta) + M_affine[1, 2] * np.cos(theta)) / s0
theta = np.pi * theta / 180 # se necesita pasar el angúlo de rotación a radianes

# implementación de la matriz de similitud vista en clase
M_sim = np.float32([[s0 * np.cos(theta), -np.sin(theta), x0],
                    [np.sin(theta), s1 * np.cos(theta), x1]])
image_similarity = cv2.warpAffine(image1, M_sim, image1.shape[:2])
cv2.imshow("Image Similarity", image_similarity)

# se calcula el error cuadratico medio para la imagen transformada con la matriz de similitud respecto
# a ala imagen de lena_warped.png
ECM_gauss_similarity = math.sqrt(np.square(np.subtract(image_similarity, image2)).mean())
print(ECM_gauss_similarity)

cv2.waitKey(0)
