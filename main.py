import functions as fn
import functions2 as fn2
import pandas as pd
import cv2

from functions import loadImage, ImagetoList
from knn import KNN
from sklearn import datasets


# load info

digits = datasets.load_digits()

"""print("Data:")
print(digits["data"]) # matriz plana
print("Target:")
print(digits["target"]) # los numeros que representan
print("Images:")
print(digits["images"]) # matriz representativa
"""
data = digits["data"]
target = digits["target"]
images = digits["images"]

img_prueba_1 = loadImage(1)
img_prueba_1_lista = ImagetoList(img_prueba_1)

amount = 10
totalDigits = len(digits["target"])

print("Hallando las matrices promedio...")

# create an empty matrix
numbers = fn.createMatrix(amount)

# create an empty list
meanNumbers = [None] * amount

# each cell has its own image numbers
for i in range(totalDigits):
    numbers[target[i]].append(images[i])

# calculate average matrix
for i in range(amount):
    meanNumbers[i] = fn.meanMatrix(numbers[i])



print("\n...\n")
print("Matrices promedios calculadas con éxito!")

# for i in range(10):
#     fn.showMatrix(i, meanNumbers)

# idx = int(input("Ingresa el numero de la matriz que quieres ver (0 para salir): "))
#
# while(idx!=0):
#     if (idx < 0 or idx >= amount):
#         idx = int(input("Ingresa el numero de la matriz que quieres ver (0 para salir): "))
#     else:
#         fn.showMatrix(i, meanNumbers)


# print(meanNumbers)


number = int(input("Ingresa el numero de la imagen que deseas analizar: "))
img = fn.loadImage(number)

knn = KNN(img, 3)

print("Los 3 knn son")
print(knn.findKNeighbors(data, target))


'''
# -----------------F---------------

# ----------------G---------------
# Transforming meanNumbers into lists
mean_numbers_in_list = []
for a in range(10):
    meanNumber = []
    for i in range(len(meanNumbers[0])):
        for j in range(len(meanNumbers[0][0])):
            meanNumber.append(meanNumbers[a][i][j])
    mean_numbers_in_list.append(meanNumber)

knn = KNN(img, 3)
print("Los 3 vecinos más cercanos son: ",knn.findKNeighbors(data, target))
img = ImagetoList(img)
digit_prediction_1 = int(fn2.findKNeighbors2(img, data, target))
print("Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número ",digit_prediction_1)
digit_prediction_2 = int(fn2.distancia_hacia_promedios(img, mean_numbers_in_list))
print("Soy la inteligencia artificial versión 2, y he detectado que el dígito ingresado corresponde al número ",digit_prediction_2)

# -----------------------H-------------------------
def resultados_generales():
    print("-----RESULTADOS-----")
    print(f"{'Dígito Introducido':<20} {'Predicción 1':<14} {'Predicción 2':<14}")
    print("-" * 50)  # Línea separadora

    for i in range(10):
        img = ImagetoList(loadImage(i))
        digit_prediction_1 = int(fn2.findKNeighbors2(img, data, target))
        digit_prediction_2 = int(fn2.distancia_hacia_promedios(img, mean_numbers_in_list))

        # Imprimir fila con las predicciones
        print(f"{i:<20} {digit_prediction_1:<14} {digit_prediction_2:<14}")


rpta = input("Mostrar los resultados generales? ")
if rpta == "y":
    resultados_generales()

# print("Digito: ",i)
    # print("Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número ",
    #       digit_prediction_1)
    # print("Soy la inteligencia artificial versión 2, y he detectado que el dígito ingresado corresponde al número ",
    #       digit_prediction_2)


