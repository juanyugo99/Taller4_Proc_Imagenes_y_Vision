import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from sklearn.utils import shuffle

# metodo para recrear la imagen ingresada aplicando la clasificación del metodo
def recreate_image(centers, labels, rows, cols):
    d = centers.shape[1]
    image_clusters = np.zeros((rows, cols, d))
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            image_clusters[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image_clusters

# metodo para encontrar la suma de las distancias del color de cada pixel a su cluster correspondiente
def intraclusterdist(image, centers, labels, rows, cols):
    dist = 0
    label_idx = 0
    for i in range(rows):
        for j in range(cols):
            centroid = centers[labels[label_idx]]
            point = image[i,j,:]
            dist += np.sqrt(np.power(point[0] - centroid[0], 2) + np.power(point[1] - centroid[1], 2) + np.power(point[2] - centroid[2], 2))
            label_idx += 1
    return dist

# se pide al usuario que ingrese la direccion de la imagne y que seleccione que metodo de clasificacion desea probar
path = input('Inserte la ruta de la imagen: ')
select = int(input('Escriba (0) para usar Kmeans o (1) para usar GMM: '))
method = ['kmeans', 'gmm']
image = cv2.imread(path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ajustes preliminares para la implementacion de los algoritmos de clasisficación
image = np.array(image, dtype=np.float64) / 255
rows, cols, ch = image.shape
assert ch == 3
image_array = np.reshape(image, (rows * cols, ch))

# se muestra la figura original antes de ser modificada por el algoritmo de clasificación seleccionado
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image')
plt.imshow(image)

distance = [] # en esta lista se guardaran las distancias entre clusters de cara iteracion

mostrar = input('Desea observar los resultados de clasificacion para cada iteracion?\n esto puede realentizar el script (si/no): ').lower()
fig_count = 1
# iteracion aumentando el numero de clusters de clasificación de colores
for ncolors in range(1,11):

    # dependiendo de que metodo se seleccione realiza el algoritmo correspondiente
    image_array_sample = shuffle(image_array, random_state=0)[:10000]
    if method[select] == 'gmm':
        model = GMM(n_components=ncolors).fit(image_array_sample)
        labels = model.predict(image_array)
        centers = model.means_
        dist = intraclusterdist(image, centers, labels, rows, cols) # se calcula la distancia intra-clusters
        distance.append(dist) # se adjunta el resultado a la lista de distancias

    else:
        model = KMeans(n_clusters=ncolors, random_state=0).fit(image_array_sample)
        labels = model.predict(image_array)
        centers = model.cluster_centers_
        dist = intraclusterdist(image, centers, labels, rows, cols) # se calcula la distancia intra-clusters
        distance.append(dist) # se adjunta el resultado a la lista de distancias

    if mostrar == 'si':
        # se muestra el resultado de la clasificacion para cada iteración
        fig_count += 1
        plt.figure(fig_count)
        plt.clf()
        plt.axis('off')
        plt.title('Quantized image ({} colors, method={})'.format(ncolors, method[select]))
        plt.imshow(recreate_image(centers, labels, rows, cols))

# se muestran como cambian las distancias intra-clusters por cada iteracion del codigo
fig_count += 1
plt.figure(fig_count)
plt.clf()
plt.title('intra-cluster distance')
plt.xlabel('N Colors')
plt.ylabel('d')
plt.plot(distance)
plt.show()