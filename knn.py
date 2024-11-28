
class KNN:

    def __init__(self, img, k):

        self.k = k
        self.distances = []
        print(img)
        self.img = [j for i in img for j in i]


    def findKNeighbors(self, data, target):

        for i in range(len(data)):
            value = self.__euclideanDistance(data[i])
            self.distances.append((value, target[i]))

        self.distances = sorted(self.distances)

        first, second, third = self.distances[0][1], self.distances[1][1], self.distances[2][1]

        return [int(first), int(second), int(third)]

    def findKNeighbors2(self, data, target):

        distances = []

        for i in range(len(data)):
            value = self.__euclideanDistance(data[i])
            distances.append((value, target[i]))

        distances = sorted(distances)
        lista_neighbors = [distances[i][0] for i in range(len(distances))]

        return lista_neighbors

    def __euclideanDistance(self, matrix):
        acum = 0
        for index in range(0, len(self.img)):
            acum += (self.img[index] - matrix[index]) ** 2
        return int(acum ** 0.5)
