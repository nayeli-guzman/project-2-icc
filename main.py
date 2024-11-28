import functions as fn
import pandas as pd
import cv2
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

amount = 10
totalDigits = len(digits["target"])

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

for i in range(amount): # should be amount
    fn.showMatrix(i, meanNumbers)

print(meanNumbers)


# falta tener todos los png's en assests
number = int(input("Ingresa el numero de la imagen que deseas analizar: "))

img =fn.loadImage(number)

knn = KNN(img, 3)

print(knn.findKNeighbors(data, target))



# temporal

df = pd.DataFrame(data=meanNumbers[0])
separador = pd.DataFrame(data=[[0, 0, 0, 0, 0, 0, 0 , 0]])
df = pd.concat([df, separador], ignore_index=True)

for i in range(1,amount):
 dft = pd.DataFrame(data=meanNumbers[i])
 df = pd.concat([df, dft, separador], ignore_index=True)


df.to_csv("numbers.csv")


# p3



"""miMatriz = cv2.imread("C:/Users/nayel/Desktop/utec/ciclo/ICC/project-2-icc/assets/numero_de_prueba.png", cv2.IMREAD_GRAYSCALE)
imagen_pequena = cv2.resize(miMatriz, (8, 8))
print(imagen_pequena)

for i in range(8):
    for j in range(8):
        imagen_pequena[i][j] = imagen_pequena[i][j]*255/16"""