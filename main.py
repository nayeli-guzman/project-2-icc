import functions as fn
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

for i in range(amount):
    plt.figure(figsize=(3, 3))  # Tamaño de la figura
    sns.heatmap(meanNumbers[i], annot=False, cmap="Blues", cbar=True)
    plt.title(f"Matriz promedio de: {i}")
    plt.show()


print(meanNumbers)


# temporal

df = pd.DataFrame(data=meanNumbers[0])
separador = pd.DataFrame(data=[[0, 0, 0, 0, 0, 0, 0 , 0]])
df = pd.concat([df, separador], ignore_index=True)

for i in range(1,amount):
 dft = pd.DataFrame(data=meanNumbers[i])
 df = pd.concat([df, dft, separador], ignore_index=True)


df.to_csv("numbers.csv")