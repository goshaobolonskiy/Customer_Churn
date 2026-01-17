import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + y ** 2

def gradient(x, y):
    return np.array([2*x, 2*y])

# Данные, по которым строим график
x = np.linspace(-4, 4, 20)
y = np.linspace(-4, 4, 20)
X, Y = np.meshgrid(x,y)
Z = f(X,Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Подписи полей
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z = f(x,y)')
plt.title('Градиентный спуск для f(x,y) = x² + y²')

# Ограничение области просмотра графика
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 35)

# Сетка на графике
plt.grid(True, alpha=0.3)

# Функция f(x,y)
ax.plot_surface(X, Y, Z, label='f(x,y) = x² + y²', alpha=0.5, cmap='viridis', edgecolor='none', antialiased=True)


# Точка, от которой будем запускать градиентный спуск
point = [4,4, f(4,4)]
ax.scatter(*point, color='red', alpha=1, label='Начальная точка', s=50)

# Функция производной f(x)
# plt.plot(x,derivative_f(x), label='Функция производной')
# grad = gradient(X,Y)
# ax.plot_surface(grad[0], grad[1], Z)

# Градиентный спуск
x_old = point[0]
y_old = point[1]
speed = 0.1 # Шаг обучения
for i in range(20):
    grad = gradient(x_old, y_old)
    x_new = x_old - speed * grad[0]
    y_new = y_old - speed * grad[1]

    ax.scatter(x_new, y_new, f(x_new, y_new), color='orange', alpha=1,)

    x_old = x_new
    y_old = y_new

ax.scatter(x_old, y_old, f(x_old, y_old), color="red", alpha=1, s=50) # Точка минимума

# Уравнение касательной:
grad = gradient(point[0], point[1])
X = X * 0.4 + point[0]
Y = Y * 0.4 + point[0]
Z = grad[0] * (X-point[0]) + grad[1] * (Y-point[1]) + point[2]
ax.plot_surface(X[:],Y[:],Z[:], linestyle='--', label='Касательная плоскость', alpha=0.3)

ax.quiver(0, 0, 40,
          x_old, y_old, -35 + f(x_old, y_old), color='red', alpha=1, linewidth=2, arrow_length_ratio=0.1)

plt.legend(loc='best')
plt.tight_layout()
plt.show()