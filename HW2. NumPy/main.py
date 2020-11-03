import numpy as np


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = np.genfromtxt(url,dtype=object, delimiter=',')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

print(iris.itemsize)
print(iris.size)
#print(iris)



# 1. Извлечь колонку ‘species’
iris = iris[ : , :4]
#print(iris)

