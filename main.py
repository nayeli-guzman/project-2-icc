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

# for i in range(1):
#     fn.showMatrix(i, meanNumbers)

# idx = int(input("Ingresa el numero de la matriz que quieres ver (0 para salir): "))
#
# while(idx!=0):
#     if (idx < 0 or idx >= amount):
#         idx = int(input("Ingresa el numero de la matriz que quieres ver (0 para salir): "))
#     else:
#         fn.showMatrix(i, meanNumbers)


# print(meanNumbers)


# falta tener todos los png's en assests
# number = int(input("Ingresa el numero de la imagen que deseas analizar: "))
#
# img =fn.loadImage(number)
#
# knn = KNN(img, 3)
# print(knn.findKNeighbors(data, target))



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
# Calculando distancia

#-----------------------H-------------------------
for i in range(10):
    img = ImagetoList(loadImage(i))
    digit_prediction_1 = fn2.findKNeighbors2(img, data, target)
    digit_prediction_2 = fn2.distancia_hacia_promedios(img, mean_numbers_in_list)
    print("Digito: ",i)
    print("Soy la inteligencia artificial, y he detectado que el dígito ingresado corresponde al número ",
          digit_prediction_1)
    print("Soy la inteligencia artificial versión 2, y he detectado que el dígito ingresado corresponde al número ",
          digit_prediction_2)
# print(meanNumbers[1])
# print(mean_numbers_in_list[1])


# 

# Clasificación del nuevo dígito

# knn.findKNeighbors2(data,target)

# temporal

df = pd.DataFrame(data=meanNumbers[0])
separador = pd.DataFrame(data=[[0, 0, 0, 0, 0, 0, 0 , 0]])
df = pd.concat([df, separador], ignore_index=True)

for i in range(1,amount):
 dft = pd.DataFrame(data=meanNumbers[i])
 df = pd.concat([df, dft, separador], ignore_index=True)


# df.to_csv("numbers.csv")


# p3



"""miMatriz = cv2.imread("C:/Users/nayel/Desktop/utec/ciclo/ICC/project-2-icc/assets/numero_de_prueba.png", cv2.IMREAD_GRAYSCALE)
imagen_pequena = cv2.resize(miMatriz, (8, 8))
print(imagen_pequena)

for i in range(8):
    for j in range(8):
        imagen_pequena[i][j] = imagen_pequena[i][j]*255/16"""

