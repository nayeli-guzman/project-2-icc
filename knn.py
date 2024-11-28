
class KNN:

    def __init__(self, img, k):

        self.k = k
        print(img)
        self.img = [j for i in img for j in i]


    def findKNeighbors(self, data, target):

        distances = []

        for i in range(len(data)):
            value = self.__euclideanDistance(data[i])
            distances.append((value, target[i]))

        distances = sorted(distances)

        first, second, third = distances[0][1], distances[1][1], distances[2][1]

        return [int(first), int(second), int(third)]

    def __euclideanDistance(self, matrix):
        acum = 0
        for index in range(0, len(self.img)):
            acum += (self.img[index] - matrix[index]) ** 2
        return int(acum ** 0.5)
