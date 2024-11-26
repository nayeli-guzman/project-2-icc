
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


