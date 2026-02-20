import random


def beautiful_print_matr(lst):
    for row in lst:
        print(row)


def rand_matr(n, m):
    return [[random.randint(0, 10) for _ in range(m)] for _ in range(n)]


def minor(matr, i, j):
    # Удаляем i-ую строку
    matr_without_row = matr[:i] + matr[i + 1:]
    # Удаляем j-ый столбец из каждой строки
    new_matr = [row[:j] + row[j + 1:] for row in matr_without_row]
    return determinant(new_matr)


def algebraic_complement(matr, i, j):
    return minor(matr, i, j) * (-1) ** (i + j)


def transposition(matr):
    # Если это вектор, преобразуем в матрицу-столбец
    if matr and not isinstance(matr[0], list):
        return [[x] for x in matr]

    rows, cols = len(matr), len(matr[0])
    return [[matr[i][j] for i in range(rows)] for j in range(cols)]


def multi_matr_matr(a, b):
    if not a or not b:
        raise ValueError('Пустые матрицы')

    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])

    if cols_a != rows_b:
        raise ValueError(f'Нельзя умножить матрицу {rows_a}x{cols_a} на {rows_b}x{cols_b}')

    result = [[0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]

    return result


def multi_scalar_matr(matr, scalar):
    return [[element * scalar for element in row] for row in matr]


def determinant(matr):
    rows = len(matr)
    if rows == 0:
        return 0

    # Проверка на квадратность
    if rows != len(matr[0]):
        raise ValueError('Матрица не квадратная')

    # Базовые случаи
    if rows == 1:
        return matr[0][0]
    if rows == 2:
        return matr[0][0] * matr[1][1] - matr[1][0] * matr[0][1]

    # Рекурсивное разложение по первой строке
    det = 0
    for j in range(rows):
        sign = (-1) ** j
        det += sign * matr[0][j] * minor(matr, 0, j)

    return det


def inverse_matr(matr):
    det = determinant(matr)
    if abs(det) < 1e-10:  # Проверка на близость к нулю
        raise ValueError('Определитель равен нулю - матрица вырождена')

    n = len(matr)
    # Строим матрицу алгебраических дополнений
    alg_matr = [[algebraic_complement(matr, i, j) for j in range(n)] for i in range(n)]
    # Транспонируем (получаем присоединенную матрицу) и умножаем на 1/det
    return multi_scalar_matr(transposition(alg_matr), 1 / det)


def solve_system_manual(A, b):
    rows, cols = len(A), len(A[0])

    if rows != cols:
        raise ValueError('Матрица системы не квадратная')

    # Преобразуем b в матрицу-столбец, если это вектор
    if b and not isinstance(b[0], list):
        b = [[x] for x in b]

    return multi_matr_matr(inverse_matr(A), b)


# Тестирование
if __name__ == "__main__":
    a = [[1, 1, 7],
         [4, 5, 8],
         [2, 1, 0]]

    b = [[5, 5, 5],
         [5, 5, 6],
         [5, 2, 5]]

    print("Матрица A:")
    beautiful_print_matr(a)
    print("\nМатрица B:")
    beautiful_print_matr(b)
    print("\nРешение системы B * x = [1,2,3]^T:")
    beautiful_print_matr(solve_system_manual(b, [1, 2, 3]))

    print(f"\nОпределитель A: {determinant(a)}")
    print("Обратная матрица A:")
    beautiful_print_matr(inverse_matr(a))
