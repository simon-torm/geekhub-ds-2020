import numpy as np


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris = np.genfromtxt(url,dtype=object, delimiter=',')
names = ('sepallength', 'sepalwidth', 'petallength', 'petalwidth', 'species')

#print(iris.itemsize)
#print(iris.size)
#print(iris)



# 1.Извлечь колонку ‘species’
print('\nTASK #1')
#не знаю, чи варто було таким шляхом робити. тут звісно можна було вказати індекс "4"
#але я віришив, що нам не відомо індекс потрібної колонки за умовою
iris = np.delete(iris, np.where(np.asarray(names) == 'species'), 1)
print(iris)

# 2.Преобразовать первые 4 колонки в 2D массив?
print('\nTASK #2')
#iris = np.array(iris, dtype = np.float64)
iris = iris.astype(np.float64)
#print(iris)

# 3.Посчитать mean, median, standard deviation по 1-й колонке
print('\nTASK #3')
iris_mean = iris[ : , 0].mean()
print('mean =', iris_mean)
iris_median = np.median(iris[ : , 0])
print('median =', iris_median)
iris_std = iris[ : , 0].std()
print('standard deviation =', iris_std)

# 4.Вставить 20 значений np.nan на случайные позиции в массиве
print('\nTASK #4')
rands_idx = np.random.choice(range(iris.size), 20, replace=False)
rands_idx = np.unravel_index(rands_idx, iris.shape)
iris[rands_idx] = np.nan

# 5.Найти позиции вставленных значений np.nan в 1-й колонке
print('\nTASK #5')
pos_nan = np.where(np.isnan(iris[ : , 0]))
print('pos_nan =', pos_nan)

# 6. Отфильтровать массив по условию значения в 3-й колонке > 1.5 и значения в 1-й колонке < 5.0
print('\nTASK #6')
mas_filter = np.where((iris[ : , 2] > 1.5) & (iris[ : , 0] < 5))
res_mas_filter = iris[mas_filter, : ]
print('res_filter:\n', res_mas_filter)

# 7.Заменить все значения np.nan на 0
print('\nTASK #7')
iris[np.isnan(iris)] = 0
#print(iris)

# 8.Посчитать все уникальные значения в массиве и вывести их вместе с подсчитанным количеством
print('\nTASK #8')
(unique_list, unique_counts) = np.unique(iris, return_counts=True)
print('value - count')
for un, count in zip(unique_list, unique_counts):
    print('',un, ' - ', count)

# 9.Разбить массив по горизонтали на 2 массива
print('\nTASK #9')
top, bot = np.vsplit(iris, 2)

# 10.Отсортировать оба получившихся массива по 1-й колонке: 1-й по возрастанию, 2-й по убыванию
print('\nTASK #10')
sort_inxs1 = np.lexsort((top[:, 1], top[:, 0]))
sort_inxs2 = np.lexsort((bot[:, 1], bot[:, 0]))
top = top[sort_inxs1]
bot = np.flip(bot[sort_inxs2], 0)

# 11.Склеить оба массива обратно
print('\nTASK #11')
iris = np.concatenate([top, bot])
#print(iris)

# 12.Найти наиболее часто повторяющееся значение в массиве
print('\nTASK #12')
print('max repeated value:', unique_list[np.argmax(unique_counts)])

# 13.Написать функцию, которая бы умножала все значения в колонке, меньше среднего значения в этой колонке, на 2, и делила остальные значения на 4. Применить к 3-й колонке
print('\nTASK #13')


def my_func(a):
    a2 = a.copy()
    mean_val = a2.mean()
    cond1 = np.where(a2 < mean_val)
    cond2 = np.where(a2 >= mean_val)
    a2[cond1] = a2[cond1] * 2
    a2[cond2] = a2[cond2] / 4
    return a2


iris[ : , 2] = np.apply_along_axis(my_func, 0, iris[ : , 2])