def distanciaEuclidiana(img,lista):
    acum = 0
    for index in range(len(lista)):
        acum += (img[index] - lista[index]) ** 2
    return int(acum ** 0.5)


def findKNeighbors2(img, data, target):
    distances = []
    for i in range(len(data)):
        value = distanciaEuclidiana(img,data[i])
        distances.append((value, target[i]))

    distances = sorted(distances)
    lista_neighbors = [distances[i][1] for i in range(len(distances))]

    knn_neighbors = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    i = 0
    while True:
        knn_neighbors[lista_neighbors[i]] += 1
        if knn_neighbors[lista_neighbors[i]] > 1:
            valor_predicho = lista_neighbors[i]
            break
        i += 1
    return valor_predicho

def distancia_hacia_promedios(img2,lista):
    distance_list = []
    for i in range(10):
        distance = distanciaEuclidiana(img2,lista[i])
        distance_list.append(distance)
    menor = min(distance_list)
    return distance_list.index(menor)