import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2


def createMatrix(size : int = 10):
    matrix = [[] for _ in range(size)]
    return matrix

def meanMatrix(arrayMatrix, size:int = 8):

    response = [[0] * size for _ in range(size)]

    for matrix in arrayMatrix:
        for i in range(size):
            for j in range(size):
                response[i][j] = response[i][j] + matrix[i][j]
    
    response = [[int(response[i][j] // len(arrayMatrix)) for j in range(size)] for i in range(size)]

    return response

def showMatrix(idx:int, matrix):
    plt.figure(figsize=(3, 3))
    sns.heatmap(matrix[idx], annot=False, cmap="Blues", cbar=True)
    plt.title(f"Matriz promedio de: {idx}")
    plt.show()

def loadImage(number : int):

    path = f"assets/{number}.png"
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    newImg = cv2.resize(img, (8, 8))

    for y in range(8):
        for x in range(8):
            newImg[y][x] = (255 - newImg[y][x]) / 255*16

    return newImg

def ImagetoList(img):
    img_en_lista = []
    for i in range(len(img)):
        for j in range(len(img[0])):
            img_en_lista.append(int(img[i][j]))
    return img_en_lista

