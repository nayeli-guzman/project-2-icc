
class KNN:

    def __init__(self, img, k):

        self.k = k
        self.distances = []
        print(img)
        self.img = [j for i in img for j in i]


    def findKNeighbors(self, data, target):

        self.distances = []

        for i in range(len(data)):
            value = self.euclideanDistance(data[i])
            self.distances.append((value, target[i]))

        self.distances = sorted(self.distances)

        first, second, third = self.distances[0][1], self.distances[1][1], self.distances[2][1]

        return [int(first), int(second), int(third)]

    def findKNeighbors2(self, data, target):

        self.distances = []

        for i in range(len(data)):
            value = self.euclideanDistance(data[i])
            self.distances.append((value, target[i]))

        self.distances = sorted(self.distances)
        lista_neighbors = [self.distances[i][1] for i in range(len(self.distances))]

        knn_neighbors = {}
        i = 0
        while True:
            knn_neighbors[lista_neighbors[i]] += 1
            if knn_neighbors[lista_neighbors[i]] > 1:
                print("Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número ", lista_neighbors[i])
                break
            i += 1
        return None

    def euclideanDistance(self, matrix):
        acum = 0
        for index in range(0, len(self.img)):
            acum += (self.img[index] - matrix[index]) ** 2
        return int(acum ** 0.5)
