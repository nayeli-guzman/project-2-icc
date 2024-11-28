import functions as fn
from knn import KNN
from sklearn import datasets

import pandas as pd

# load info
digits = datasets.load_digits()

"""
print("Data:")
print(digits["data"]) # matriz plana
print("Target:")
print(digits["target"]) # los numeros que representan
print("Images:")
print(digits["images"]) # matriz representativa
"""

data = digits["data"]
target = digits["target"]
images = digits["images"]

amount = 10 # [0,1,  ... , 9]
totalDigits = len(digits["target"])

print("Hallando las matrices promedio...")

# matriz para almacenar todas las n matrices  
# que representan al número i en el indice i
numbers = fn.createMatrix(amount)

# lista donde se almacenarán las 
# matrices promedio
meanNumbers = [None] * amount

for i in range(totalDigits):
    numbers[target[i]].append(images[i])

for i in range(amount):
    meanNumbers[i] = fn.meanMatrix(numbers[i])

print("\n...\n")
print("Matrices promedios calculadas con éxito!")

for i in range(amount):
    fn.showMatrix(i, meanNumbers)

"""
idx = int(input("Ingresa el numero de la matriz que quieres ver (0 para salir): "))

while(idx!=0):
    if (idx < 0 or idx >= amount):
        idx = int(input("Ingresa el numero de la matriz que quieres ver (0 para salir): "))
    else:
        fn.showMatrix(idx, meanNumbers)
"""

# print(meanNumbers)


# falta tener todos los png's en assests
number = int(input("Ingresa el numero de la imagen que deseas analizar: "))

img =fn.loadImage(number)

knn = KNN(img, 3)

# print(knn.findKNeighbors(data, target))

# Clasificación del nuevo dígito

knn.findKNeighbors2(data,target)

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