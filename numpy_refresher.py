import numpy as np
import random


def beutiful_print_matr(lst: list) -> None:
    for i in lst:
        print(i)


def rand_matr(n: int, m: int) -> list:
    lst: list = []
    for i in range(n):
        lst2 = []
        for j in range(m):
            lst2.append(random.randint(0,10))
        lst.append(lst2)
    return lst


def mean_matr_col(lst: list, col: int) -> float:
    mean = 0

    for i in lst:
        mean += i[col]
    return mean / len(lst)

def standard_deviation(lst: list) -> float:
    mean = 0
    for i in range(len(lst)):
        for j in lst[i]:
            mean += j
    mean /= len(lst) * len(lst[0])
    dev = 0
    for i in range(len(lst)):
        for j in lst[i]:
            dev += (mean - j) ** 2
    dev /= len(lst) * len(lst[0])
    dev **= 0.5
    return dev



lst = rand_matr(4, 3)
print(lst)
a = np.array(lst)
print(a)
# print(np.mean(a))
print(np.std(a, axis=1))
print(sum(np.std(a, axis=1)) / 3)
print(standard_deviation(lst))
# print(mean_matr_col(lst, 0), mean_matr_col(lst, 1), mean_matr_col(lst, 2))

# beutiful_print_matr(rand_list(4,3))