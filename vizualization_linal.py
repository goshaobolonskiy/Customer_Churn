import numpy as np
import matplotlib.pyplot as plt

# ========== Меняем эту матрицу ==========
matrix = np.array([[1, 0],  # Растяжение по X
                   [0, 1]])

# ========== Основной код ==========

# Создаем фигуру
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Базисные векторы
i = np.array([1, 0])  # красный
j = np.array([0, 1])  # синий

# 1. График слева: до
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')
ax1.set_title('До трансформации')

# Сетка
for x in range(-2, 3):
    ax1.plot([x, x], [-2, 2], 'gray', alpha=0.2)
for y in range(-2, 3):
    ax1.plot([-2, 2], [y, y], 'gray', alpha=0.2)

# Базисные векторы
ax1.arrow(0, 0, i[0], i[1], color='red', width=0.02, head_width=0.1)
ax1.arrow(0, 0, j[0], j[1], color='blue', width=0.02, head_width=0.1)

# 2. График справа: после
# Преобразуем векторы
i_new = matrix @ i
j_new = matrix @ j

ax2.set_aspect('equal')
ax2.set_title(f'После трансформации')

# Преобразованная сетка
for x in range(-2, 3):
    for y in range(-2, 3):
        point = np.array([x, y])
        point_new = matrix @ point
        # Соседние точки для линий
        if x < 2:
            point_right = np.array([x + 1, y])
            point_right_new = matrix @ point_right
            ax2.plot([point_new[0], point_right_new[0]],
                     [point_new[1], point_right_new[1]], 'gray', alpha=0.2)
        if y < 2:
            point_up = np.array([x, y + 1])
            point_up_new = matrix @ point_up
            ax2.plot([point_new[0], point_up_new[0]],
                     [point_new[1], point_up_new[1]], 'gray', alpha=0.2)

# Преобразованные векторы
ax2.arrow(0, 0, i_new[0], i_new[1], color='red', width=0.02, head_width=0.1)
ax2.arrow(0, 0, j_new[0], j_new[1], color='blue', width=0.02, head_width=0.1)

plt.suptitle(f'Матрица: [{matrix[0, 0]}, {matrix[0, 1]}], [{matrix[1, 0]}, {matrix[1, 1]}]')
plt.tight_layout()
plt.show()
